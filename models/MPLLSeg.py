import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from torch.nn import TransformerDecoderLayer, LayerNorm, TransformerDecoder


class GlobalZ(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.query = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.query_bn = nn.GroupNorm(1, in_channel // 2)
        self.key = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.key_bn = nn.GroupNorm(1, in_channel // 2)
        self.value = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.value_bn = nn.GroupNorm(1, in_channel)

    def forward(self, query, key, value):
        b, n, c, h, w = query.shape
        query = query.reshape(b * n, c, h, w)
        key = key.reshape(b * n, c, h, w)
        value = value.reshape(b * n, c, h, w)

        query = self.query(query)
        query = self.query_bn(query)
        key = self.key(key)
        key = self.key_bn(key)
        value = self.value(value)
        value = self.value_bn(value)

        query_pool_g = F.adaptive_avg_pool2d(query, (1, 1))
        query_pool_g = query_pool_g.view(b, n, c // 2)
        key_pool_g = F.adaptive_avg_pool2d(key, (1, 1))
        key_pool_g = key_pool_g.view(b, n, c // 2)
        sim_slice = torch.einsum('bmd,bnd->bmn', query_pool_g, key_pool_g)
        sim_slice = sim_slice / (c // 2) ** 0.5
        sim_slice = torch.softmax(sim_slice, dim=-1)
        context_pool_slice = torch.einsum('bmn,bnchw->bmchw', sim_slice, value.reshape(b, n, c, h, w))

        return context_pool_slice


class GlobalS(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.query = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.query_bn = nn.GroupNorm(1, in_channel // 2)
        self.key = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.key_bn = nn.GroupNorm(1, in_channel // 2)
        self.value = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.value_bn = nn.GroupNorm(1, in_channel)

    def forward(self, query, key, value):
        b, n, c, h, w = query.shape
        query = query.reshape(b * n, c, h, w)
        key = key.reshape(b * n, c, h, w)
        value = value.reshape(b * n, c, h, w)

        query = self.query(query)
        query = self.query_bn(query)
        key = self.key(key)
        key = self.key_bn(key)
        value = self.value(value)
        value = self.value_bn(value)

        query_pool_g = query.reshape(b, n, c // 2, h, w)
        key_pool_g = key.reshape(b, n, c // 2, h, w)
        # value = value.reshape(b, n, c, h, w)

        query_pool_g = torch.mean(query_pool_g, 1).reshape(b, c // 2, h * w)
        key_pool_g = torch.mean(key_pool_g, 1).reshape(b, c // 2, h * w)

        sim_slice = torch.einsum('bci,bcj->bij', query_pool_g, key_pool_g)
        sim_slice = sim_slice / (c // 2) ** 0.5
        sim_slice = torch.softmax(sim_slice, dim=-1)
        context_pool_s = torch.einsum('bij,bncj->bnci', sim_slice, value.reshape(b, n, c, h * w))
        context_pool_s = context_pool_s.reshape(b, n, c, h, w)

        return context_pool_s


class TokenLearnerModuleV2(nn.Module):
    def __init__(self, in_channel=512, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.ln = nn.GroupNorm(1, in_channel)
        self.global_z = GlobalZ(in_channel)
        self.global_s = GlobalS(in_channel)
        # self.conv1 = nn.Conv2d(in_channel, in_channel, (1, 1), groups=8, bias=False)
        self.conv_3x3 = nn.Conv3d(in_channel, in_channel, (3, 3, 3), groups=8, bias=False, padding=(1, 1, 1))
        # self.conv_5x5 = nn.Conv3d(in_channel, in_channel, (5, 5, 5), groups=8, bias=False, padding=(2, 2, 2))
        # self.conv_7x7 = nn.Conv3d(in_channel, in_channel, (7, 7, 7), groups=8, bias=False, padding=(3, 3, 3))
        self.conv2 = nn.Conv2d(in_channel, num_tokens, (1, 1), bias=False)
        self.conv_feat = nn.Conv2d(in_channel, in_channel, (1, 1), groups=8, bias=False)

    def forward(self, x, x_aux):
        b, n, c, h, w = x.shape
        global_z = self.global_z(x, x_aux, x_aux)
        global_s = self.global_s(x, x_aux, x_aux)
        local = self.conv_3x3(x_aux.permute(0, 2, 1, 3, 4))
        local = torch.sigmoid(local * x.permute(0, 2, 1, 3, 4)) * local
        local = local.permute(0, 2, 1, 3, 4)

        selected = global_z + global_s + local + x
        selected = self.ln(selected.reshape(b * n, c, h, w))
        feat = selected
        selected = self.conv2(selected)
        selected = torch.softmax(selected.reshape(-1, self.num_tokens, h * w), -1)

        feat = self.conv_feat(feat)
        feat = feat.reshape(-1, c, h * w)
        feat = torch.einsum("bts, bcs->btc", selected, feat)
        return feat


class TokenFuser(nn.Module):
    def __init__(self, in_channel=512, num_tokens=8):
        super().__init__()
        self.ln1 = nn.GroupNorm(1, in_channel)
        self.ln2 = nn.GroupNorm(1, in_channel)
        self.ln3 = nn.GroupNorm(1, in_channel)
        self.conv1 = nn.Linear(num_tokens, num_tokens)
        self.mix = nn.Conv2d(in_channel, num_tokens, (1, 1), groups=8, bias=False)

    def forward(self, tokens, origin):
        # tokens = tokens.permute(0, 2, 1)
        tokens = self.ln1(tokens)
        tokens = self.conv1(tokens)
        tokens = self.ln2(tokens)
        # tokens = tokens.permute(0, 2, 1)

        origin = self.ln3(origin)
        origin = self.mix(origin)
        origin = torch.sigmoid(origin)
        mix = torch.einsum("bct,bthw->bchw", tokens, origin)
        return mix


class TransformerBlock(nn.Module):
    def __init__(self, in_channel=512,
                 n_ctx=9,
                 num_tokens=8):
        super().__init__()
        self.n_ctx = n_ctx
        self.num_tokens = num_tokens
        self.in_channel = in_channel
        # self.ln1 = nn.GroupNorm(1, in_channel)
        # self.ln2 = nn.GroupNorm(1, in_channel)
        # self.ln3 = nn.GroupNorm(1, in_channel)
        # self.mha1 = nn.MultiheadAttention(embed_dim=in_channel,
        #                                   num_heads=8,
        #                                   dropout=0.1,
        #                                   bias=False)
        decoder_layer = TransformerDecoderLayer(in_channel, nhead=8, dim_feedforward=1024, dropout=0.1,
                                                batch_first=True)
        decoder_norm = LayerNorm(in_channel)
        self.TokenLearner_a = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_v = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.decoder_a = TransformerDecoder(decoder_layer, 1, decoder_norm)
        self.decoder_v = TransformerDecoder(decoder_layer, 1, decoder_norm)
        self.TokenFuser = TokenFuser(in_channel, num_tokens=num_tokens)

    def forward(self, x):
        x_a, x_v = x[:, 0], x[:, 1]
        b, n, c, h, w = x_a.shape
        token_a = self.TokenLearner_a(x_a, x_v)
        token_v = self.TokenLearner_v(x_v, x_a)

        token_a = token_a.reshape(b, n * self.num_tokens, c)
        token_v = token_v.reshape(b, n * self.num_tokens, c)

        token_a_tmp = self.decoder_a(token_a, torch.cat([token_a, token_v], 1))
        token_v_tmp = self.decoder_v(token_v, torch.cat([token_v, token_a], 1))

        token_a_tmp = token_a_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_v_tmp = token_v_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)

        x_a = x_a.reshape(b * n, c, h, w)
        x_v = x_v.reshape(b * n, c, h, w)
        x_a = self.TokenFuser(token_a_tmp, x_a) + x_a
        x_v = self.TokenFuser(token_v_tmp, x_v) + x_v
        # x_a = self.ln2(x_a)
        # x_v = self.ln3(x_v)
        out = torch.stack([x_a, x_v], 1).reshape(b, -1, n, c, h, w)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes, high_level_inplanes, low_level_inplanes, BatchNorm):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Sequential(
            nn.Conv2d(high_level_inplanes + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3,
                      padding=1)
            )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        # print(low_level_feat.size())
        # print(x.size())
        # x = self._modules['upsampling%dx' % 2**self.n_upsamples](x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DualSeg(nn.Module):
    def __init__(self, encoder_name="resnet34",
                 in_channels=3,
                 encoder_depth=5,
                 num_tokens=8,
                 cls=4,
                 n_context=9,
                 bn=nn.BatchNorm2d
                 ):
        super().__init__()
        self.n_context = n_context
        self.num_tokens = num_tokens
        self.num_classes = cls
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights="imagenet",
            output_stride=16,
        )
        self.decoder = Decoder(cls, 512, 64, bn)

        self.TransformerBlock = TransformerBlock(512, n_ctx=n_context, num_tokens=num_tokens)

    def forward(self, x):
        b, s, n, c, h, w = x.shape
        x = x.reshape(b * s * n, c, h, w)
        # x_v = x.reshape(b * s * n, c, h, w)
        x_feat = self.encoder(x)
        # x_v_feat = self.encoder(x_v)
        h_feat, low_feat = x_feat[-1], x_feat[2]
        h_feat = h_feat.reshape(b, s, n, *h_feat.shape[1:])
        low_feat = low_feat.reshape(b, s, n, *low_feat.shape[1:])
        h_feat = self.TransformerBlock(h_feat)
        # h_feat = self.TransformerBlock_2(h_feat)
        feat_a = h_feat[:, 0].reshape(-1, *h_feat.shape[3:])
        feat_v = h_feat[:, 1].reshape(-1, *h_feat.shape[3:])
        x_a_masks = self.decoder(feat_a, low_feat[:, 0].reshape(-1, *low_feat.shape[3:]))
        x_v_masks = self.decoder(feat_v, low_feat[:, 1].reshape(-1, *low_feat.shape[3:]))
        # x_a_feat[-2], x_v_feat[-2] = self.TransformerBlock_2(x_a_feat[-2], x_v_feat[-2])
        # x_a_output = self.decoder(*x_a_feat)
        # x_v_output = self.decoder(*x_v_feat)
        # x_a_masks = self.segmentation_head(x_a_output)
        # x_v_masks = self.segmentation_head(x_v_output)
        x_a_masks = F.interpolate(x_a_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_v_masks = F.interpolate(x_v_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_a_masks = x_a_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_v_masks = x_v_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        # x_a_masks = x_a_masks.reshape(b, n, -1, h, w)
        # x_v_masks = x_v_masks.reshape(b, n, -1, h, w)
        # final_seg = torch.stack([x_a_masks, x_v_masks], 2)  # (b, c, s, z, h, w)

        # if self.training:
        #     return x_a_masks, final_seg
        return x_a_masks, x_v_masks # (b, c, z, h, w)
        # return x_a_masks, x_v_masks

# from utils.ct_io import load_ct
# import numpy as np
# import cv2
# import torchvision
#
# ct = load_ct("/data4/liver_CT3_Z2_ts4.0/ct_00055626.npz")
# ct = torch.from_numpy(ct[2, 65 + 3]).float()
# ct = np.clip(ct, -55, 155)
# ct = (ct + 55) / (155 + 55)
# # torchvision.utils.save_image(ct, "../log/img.png", nrow=9)
# ct = ct.numpy()
# feat_tmp = selected.reshape(b, n, 16, 32, 32)[0, 0+3]
# ct_tmp = np.uint8(np.stack([ct, ct, ct], 2) * 255)
# cv2.imwrite("../log/tokens/img-2-68.png", ct_tmp)
# # ct_tmp = np.uint8(np.stack([ct_tmp, ct_tmp, ct_tmp], 2) * 255)
# for i in range(16):
#     heatmap = feat_tmp.cpu().numpy()[i]
#     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
#     # heatmap = np.transpose(heatmap, (1, 2, 0))
#     heatmap = cv2.resize(heatmap, (512, 512))
#     # heatmap = np.transpose(heatmap, (2, 0, 1))
#     # heatmap = np.concatenate(list(heatmap), 1)
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     f_img = ct_tmp + heatmap * 0.4
#     cv2.imwrite("../log/tokens/tokenmap-2-68-%d.png" % i, f_img)

if __name__ == '__main__':
    m = DualSeg().cuda()
    a = torch.rand((10, 2, 3, 3, 256, 256)).cuda()
    # b = torch.rand((90, 8, 512)).cuda()
    # c = torch.rand((90, 8, 512)).cuda()
    m(a)
    # m = TransformerBlock().cuda()
    # a = torch.rand((10, 2, 9, 512, 16, 16)).cuda()
    # m(a)
