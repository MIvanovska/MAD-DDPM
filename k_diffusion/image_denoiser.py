from torch import nn, cat
from torch.nn import functional as F
from . import layers, utils

class ConditionedSequential(nn.Sequential):
    def forward(self, input, cond):
        for module in self:
            if isinstance(module):
                input = module(input, cond)
            else:
                input = module(input)
        return input

class ConditionedResidualBlock(nn.Module):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input, cond):
        skip = self.skip(input, cond) if isinstance(self.skip) else self.skip(input)
        return self.main(input, cond) + skip


class ResConvBlock(layers.ConditionedResidualBlock):
    def __init__(self, feats_in, c_in, c_mid, c_out, group_size=32, dropout_rate=0.):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            layers.AdaGN(feats_in, c_in, max(1, c_in // group_size)),
            nn.GELU(),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            skip=skip)


class DBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., downsample=False, self_attn=False):
        modules = [nn.Identity()]
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
        super().__init__(*modules)
        self.set_downsample(downsample)
    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = cat([input, skip], dim=1)
        return super().forward(input, cond)

    def set_downsample(self, downsample):
        self[0] = layers.Downsample2d() if downsample else nn.Identity()
        return self


class UBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., upsample=False, self_attn=False):
        modules = []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
        modules.append(nn.Identity())
        super().__init__(*modules)
        self.set_upsample(upsample)

    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = cat([input, skip], dim=1)
        return super().forward(input, cond)

    def set_upsample(self, upsample):
        self[-1] = layers.Upsample2d() if upsample else nn.Identity()
        return self


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(feats_in if i == 0 else feats_out, feats_out))
            layers.append(nn.GELU())
        super().__init__(*layers)
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)


class ImageDenoiserModelV1(nn.Module):
    def __init__(self, c_in, feats_in, depths, channels, self_attn_depths, dropout_rate=0., patch_size=1):
        super().__init__()
        self.c_in = c_in
        self.channels = channels
        self.patch_size = patch_size
        self.timestep_embed = layers.FourierFeatures(1, feats_in)
        self.mapping = MappingNet(feats_in, feats_in)
        self.proj_in = nn.Conv2d(c_in * self.patch_size ** 2, channels[0], 1)
        nn.init.zeros_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        self.proj_out = nn.Conv2d(channels[0], c_in * self.patch_size ** 2, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # from here on we define the UNet
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[max(0, i - 1)]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample=i > 0, self_attn=self_attn_depths[i], dropout_rate=dropout_rate))

        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[max(0, i - 1)]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample=i > 0, self_attn=self_attn_depths[i], dropout_rate=dropout_rate))
        self.u_net = layers.UNet(d_blocks, reversed(u_blocks), skip_stages=0)

    def forward(self, input, sigma):
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(utils.append_dims(c_noise, 2)) # embedding of the sampled level of noise
        mapping_out = self.mapping(timestep_embed) # conditioning on timestep
        cond = {'cond': mapping_out}
        if self.patch_size > 1:
            input = F.pixel_unshuffle(input, self.patch_size)
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        if self.patch_size > 1:
            input = F.pixel_shuffle(input, self.patch_size)
        return input