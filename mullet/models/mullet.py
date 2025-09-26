import os

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

class TokenLearnerModuleV2(nn.Module):
    def __init__(self, in_channel=512, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.ln = nn.GroupNorm(8, in_channel)
        self.global_z = GlobalZ(in_channel)
        self.global_s = GlobalS(in_channel)
        # self.conv1 = nn.Conv2d(in_channel, in_channel, (1, 1), groups=8, bias=False)
        self.conv_3x3 = nn.Conv3d(
            in_channel, in_channel, (3, 3, 3), groups=8, bias=False, padding=(1, 1, 1)
        )
        # self.conv_5x5 = nn.Conv3d(in_channel, in_channel, (5, 5, 5), groups=8, bias=False, padding=(2, 2, 2))
        # self.conv_7x7 = nn.Conv3d(in_channel, in_channel, (7, 7, 7), groups=8, bias=False, padding=(3, 3, 3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (1, 1), bias=False, groups=8),
            nn.GroupNorm(8, in_channel),
            nn.Mish(),
            nn.Conv2d(in_channel, in_channel, (1, 1), bias=False, groups=8),
            nn.GroupNorm(8, in_channel),
            nn.Mish(),
            nn.Conv2d(in_channel, in_channel, (1, 1), bias=False, groups=8),
            nn.GroupNorm(8, in_channel),
            nn.Mish(),
            nn.Conv2d(in_channel, num_tokens, (1, 1), bias=False, groups=8),
        )
        self.conv_feat = nn.Conv2d(in_channel, in_channel, (1, 1), groups=8, bias=False)

    def forward(self, x, x_aux):
        b, n, c, h, w = x.shape
        global_z = self.global_z(x, x_aux, x_aux)
        global_s = self.global_s(x, x_aux, x_aux)
        local = self.conv_3x3(x_aux.permute(0, 2, 1, 3, 4).contiguous())
        local = torch.sigmoid(local * x.permute(0, 2, 1, 3, 4).contiguous()) * local
        local = local.permute(0, 2, 1, 3, 4).contiguous()

        selected = global_z + global_s + local + x
        selected = self.ln(selected.reshape(b * n, c, h, w))
        feat = selected
        selected = self.conv2(selected)
        selected = torch.softmax(selected.reshape(-1, self.num_tokens, h * w), -1)

        feat = self.conv_feat(feat)
        feat = feat.reshape(-1, c, h * w)
        feat = torch.einsum("bts, bcs->btc", selected, feat).contiguous()
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
        mix = torch.einsum("bct,bthw->bchw", tokens, origin).contiguous()
        return mix


class Head(nn.Module):
    def __init__(self, in_channel=512, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_channel = in_channel
        # self.ln1 = nn.GroupNorm(1, in_channel)
        # self.ln2 = nn.GroupNorm(1, in_channel)
        # self.ln3 = nn.GroupNorm(1, in_channel)
        # self.mha1 = nn.MultiheadAttention(embed_dim=in_channel,
        #                                   num_heads=8,
        #                                   dropout=0.1,
        #                                   bias=False)
        decoder_layer = TransformerDecoderLayer(
            in_channel, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        # decoder_norm = LayerNorm(in_channel)
        self.TokenLearner_p = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_a = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_v = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_d = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.decoder_p = TransformerDecoder(decoder_layer, 6)
        self.decoder_a = TransformerDecoder(decoder_layer, 6)
        self.decoder_v = TransformerDecoder(decoder_layer, 6)
        self.decoder_d = TransformerDecoder(decoder_layer, 6)
        self.TokenFuser = TokenFuser(in_channel, num_tokens=num_tokens)
        # self._init_weight()

    def forward(self, x, b, s, n):
        bsn, c, h, w = x.shape
        x = x.reshape(b, s, n, c, h, w)
        x_p, x_a, x_v, x_d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        token_p = self.TokenLearner_p(x_p, x_v)
        token_a = self.TokenLearner_a(x_a, x_v)
        token_v = self.TokenLearner_v(x_v, x_a)
        token_d = self.TokenLearner_d(x_d, x_a)

        token_p = token_p.reshape(b, n * self.num_tokens, c)
        token_a = token_a.reshape(b, n * self.num_tokens, c)
        token_v = token_v.reshape(b, n * self.num_tokens, c)
        token_d = token_d.reshape(b, n * self.num_tokens, c)

        token_p_tmp = self.decoder_p(token_p, torch.cat([token_p, token_a, token_v], 1))
        token_a_tmp = self.decoder_a(token_a, torch.cat([token_p, token_a, token_v], 1))
        token_v_tmp = self.decoder_v(token_v, torch.cat([token_p, token_a, token_v], 1))
        token_d_tmp = self.decoder_d(token_d, torch.cat([token_p, token_a, token_v], 1))

        token_p_tmp = (
            token_p_tmp.reshape(-1, self.num_tokens, self.in_channel)
            .permute(0, 2, 1)
            .contiguous()
        )
        token_a_tmp = (
            token_a_tmp.reshape(-1, self.num_tokens, self.in_channel)
            .permute(0, 2, 1)
            .contiguous()
        )
        token_v_tmp = (
            token_v_tmp.reshape(-1, self.num_tokens, self.in_channel)
            .permute(0, 2, 1)
            .contiguous()
        )
        token_d_tmp = (
            token_d_tmp.reshape(-1, self.num_tokens, self.in_channel)
            .permute(0, 2, 1)
            .contiguous()
        )

        x_p = x_p.reshape(b * n, c, h, w)
        x_a = x_a.reshape(b * n, c, h, w)
        x_v = x_v.reshape(b * n, c, h, w)
        x_d = x_d.reshape(b * n, c, h, w)
        x_p = self.TokenFuser(token_p_tmp, x_p) + x_p
        x_a = self.TokenFuser(token_a_tmp, x_a) + x_a
        x_v = self.TokenFuser(token_v_tmp, x_v) + x_v
        x_d = self.TokenFuser(token_d_tmp, x_d) + x_d
        out = torch.stack([x_p, x_a, x_v, x_d], 1).reshape(-1, c, h, w)
        return out


class SegformerHead(nn.Module):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(
        self,
        in_channels,
        cls,
        channels=256,
        in_index=[0, 1, 2, 3],
        interpolate_mode="bilinear",
        num_tokens=24,
        **kwargs
    ):
        super().__init__()

        self.interpolate_mode = interpolate_mode
        self.num_tokens = num_tokens
        self.in_channels = in_channels
        num_inputs = len(self.in_channels)

        assert num_inputs == len(in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels[i],
                        out_channels=channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.Mish(True),
                )
            )

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                channels * num_inputs,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(channels, cls, kernel_size=1, bias=False),
        )

        self._init_weight()

    def forward(self, inputs, shapes):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            hw_shape = shapes[idx]
            outs.append(
                F.interpolate(
                    input=conv(x),
                    size=shapes[0],
                    mode=self.interpolate_mode,
                    align_corners=False,
                )
            )

        out = self.last_conv(torch.cat(outs, dim=1).contiguous())

        return out

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


class MULLET(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        in_channels=3,
        encoder_depth=5,
        cls=4,
        num_tokens=16,
        n_context=3,
        phase_id=[1, 2],
    ):
        super().__init__()
        # n_context=args.n_ctx
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.encoder_depth = encoder_depth
        self.phase_id = phase_id
        self.cls = cls

        self.encoder = get_encoder(
            self.encoder_name,
            in_channels=self.in_channels,
            depth=self.encoder_depth,
            weights="imagenet",
            output_stride=16,
        )

        self.head = Head(self.encoder.out_channels[-1], num_tokens=num_tokens)
        self.decoder = SegformerHead(
            in_channels=self.encoder.out_channels[2:],
            cls=cls,
            channels=256,
            in_index=[0, 1, 2, 3],
            num_tokens=num_tokens,
        )

    def forward(self, x):
        b, s, n, c, h, w = x.shape
        x = x.reshape(b * s * n, c, h, w)
        x_feat = self.encoder(x)
        x_feat[-1] = self.head(x_feat[-1], b, s, n)

        shapes = [i.shape[2:] for i in x_feat[2:]]
        mask = self.decoder(x_feat[2:], shapes)
        mask = F.interpolate(
            mask, size=(h, w), mode="bilinear", align_corners=False
        ).contiguous()
        mask = mask.view(b, s, n, -1, h, w).permute(1, 0, 3, 2, 4, 5).contiguous()

        return list(mask)


if __name__ == "__main__":
    a = MULLET().cuda()
    b = torch.rand((10, 3, 3, 3, 256, 256)).cuda()
    a(b)
