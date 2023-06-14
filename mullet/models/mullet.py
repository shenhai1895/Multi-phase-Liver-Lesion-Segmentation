import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from torch.nn import TransformerDecoderLayer, LayerNorm, TransformerDecoder

from mullet.nn.positional_encoding import PositionalEncoding1D, PositionalEncodingPermute2D


class GlobalZ(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.query = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.query_bn = nn.GroupNorm(1, in_channel // 2)
        self.key = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.key_bn = nn.GroupNorm(1, in_channel // 2)
        self.value = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.value_bn = nn.GroupNorm(1, in_channel)
        self.pos = PositionalEncoding1D(in_channel // 2)

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

        pos = self.pos(key_pool_g)

        sim_slice = torch.einsum('bmd,bnd->bmn', query_pool_g + pos, key_pool_g + pos)
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

        self.pos_emb = PositionalEncodingPermute2D(in_channel // 2)

    def forward(self, query, key, value):
        b, n, c, h, w = query.shape
        query = query.reshape(b * n, c, h, w)
        key = key.reshape(b * n, c, h, w)
        value = value.reshape(b * n, c, h, w)

        pos = self.pos_emb(query)

        query = self.query(query)
        query = self.query_bn(query) + pos
        key = self.key(key)
        key = self.key_bn(key) + pos
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

class Decoder(nn.Module):
    def __init__(self, num_classes, high_level_inplanes, low_level_inplanes, BatchNorm):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.Mish(inplace=True)
        self.last_conv = nn.Sequential(
            nn.Conv2d(high_level_inplanes + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.Mish(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3,
                      padding=1)
        )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
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


class TokenLearnerModuleV2(nn.Module):
    def __init__(self, in_channel=512, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.ln = nn.GroupNorm(1, in_channel)
        self.global_z = GlobalZ(in_channel)
        self.global_s = GlobalS(in_channel)
        self.conv_3x3 = nn.Conv3d(in_channel, in_channel, (3, 3, 3), groups=8, bias=False, padding=(1, 1, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, in_channel, (1, 1), bias=False),
                                   nn.GroupNorm(1, in_channel),
                                   nn.Mish(),
                                   nn.Conv2d(in_channel, in_channel, (1, 1), bias=False),
                                   nn.GroupNorm(1, in_channel),
                                   nn.Mish(),
                                   nn.Conv2d(in_channel, in_channel, (1, 1), bias=False),
                                   nn.GroupNorm(1, in_channel),
                                   nn.Mish(),
                                   nn.Conv2d(in_channel, num_tokens, (1, 1), bias=False),
                                   )
        self.conv_feat = nn.Conv2d(in_channel, in_channel, (1, 1), groups=8, bias=False)

    def forward(self, x, x_aux):
        b, n, c, h, w = x.shape
        global_z = self.global_z(x, x_aux, x_aux)
        global_s = self.global_s(x, x_aux, x_aux)
        local = self.conv_3x3(x_aux.permute(0, 2, 1, 3, 4))
        local = torch.sigmoid(local * x.permute(0, 2, 1, 3, 4)) * local
        local = local.permute(0, 2, 1, 3, 4)

        selected = x + global_z + global_s + local
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
        tokens = self.ln1(tokens)
        tokens = self.conv1(tokens)
        tokens = self.ln2(tokens)

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
        decoder_layer = TransformerDecoderLayer(in_channel, nhead=8, dim_feedforward=1024, dropout=0.1,
                                                batch_first=True)
        decoder_norm = LayerNorm(in_channel)
        self.TokenLearner_p = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_a = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_v = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_d = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.decoder_p = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.decoder_a = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.decoder_v = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.decoder_d = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.TokenFuser = TokenFuser(in_channel, num_tokens=num_tokens)

    def forward(self, x):
        x_p, x_a, x_v, x_d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        b, n, c, h, w = x_a.shape
        token_p = self.TokenLearner_p(x_p, x_v)
        token_a = self.TokenLearner_a(x_a, x_v)
        token_v = self.TokenLearner_v(x_v, x_a)
        token_d = self.TokenLearner_d(x_d, x_a)

        token_p = token_p.reshape(b, n * self.num_tokens, c)
        token_a = token_a.reshape(b, n * self.num_tokens, c)
        token_v = token_v.reshape(b, n * self.num_tokens, c)
        token_d = token_d.reshape(b, n * self.num_tokens, c)

        token_p_tmp = self.decoder_p(token_p, torch.cat([token_p, token_a, token_v, token_d], 1))
        token_a_tmp = self.decoder_a(token_a, torch.cat([token_p, token_a, token_v, token_d], 1))
        token_v_tmp = self.decoder_v(token_v, torch.cat([token_p, token_a, token_v, token_d], 1))
        token_d_tmp = self.decoder_d(token_d, torch.cat([token_p, token_a, token_v, token_d], 1))

        token_p_tmp = token_p_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_a_tmp = token_a_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_v_tmp = token_v_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_d_tmp = token_d_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)

        x_p = x_p.reshape(b * n, c, h, w)
        x_a = x_a.reshape(b * n, c, h, w)
        x_v = x_v.reshape(b * n, c, h, w)
        x_d = x_d.reshape(b * n, c, h, w)
        x_p = self.TokenFuser(token_p_tmp, x_p) + x_p
        x_a = self.TokenFuser(token_a_tmp, x_a) + x_a
        x_v = self.TokenFuser(token_v_tmp, x_v) + x_v
        x_d = self.TokenFuser(token_d_tmp, x_d) + x_d
        out = torch.stack([x_p, x_a, x_v, x_d], 1).reshape(b, -1, n, c, h, w)
        return out


class MULLET(nn.Module):
    def __init__(self, encoder_name="resnet34",
                 in_channels=3,
                 encoder_depth=5,
                 num_tokens=24,
                 cls=4,
                 n_context=3,
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
        x_feat = self.encoder(x)
        h_feat, low_feat = x_feat[-1], x_feat[2]
        h_feat = h_feat.reshape(b, s, n, *h_feat.shape[1:])
        low_feat = low_feat.reshape(b, s, n, *low_feat.shape[1:])
        h_feat = self.TransformerBlock(h_feat)

        feat_p = h_feat[:, 0].reshape(-1, *h_feat.shape[3:])
        feat_a = h_feat[:, 1].reshape(-1, *h_feat.shape[3:])
        feat_v = h_feat[:, 2].reshape(-1, *h_feat.shape[3:])
        feat_d = h_feat[:, 3].reshape(-1, *h_feat.shape[3:])

        x_p_masks = self.decoder(feat_p, low_feat[:, 0].reshape(-1, *low_feat.shape[3:]))
        x_a_masks = self.decoder(feat_a, low_feat[:, 1].reshape(-1, *low_feat.shape[3:]))
        x_v_masks = self.decoder(feat_v, low_feat[:, 2].reshape(-1, *low_feat.shape[3:]))
        x_d_masks = self.decoder(feat_d, low_feat[:, 3].reshape(-1, *low_feat.shape[3:]))

        x_p_masks = F.interpolate(x_p_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_a_masks = F.interpolate(x_a_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_v_masks = F.interpolate(x_v_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_d_masks = F.interpolate(x_d_masks, size=(h, w), mode='bilinear', align_corners=True)

        x_p_masks = x_p_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_a_masks = x_a_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_v_masks = x_v_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_d_masks = x_d_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        return x_p_masks, x_a_masks, x_v_masks, x_d_masks  # (b, c, z, h, w)