import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAdaptiveShift(nn.Module):
    def __init__(
        self, 
        channels,
        n_groups=4,
        kernel_size=1,
        dilation=1,
        offset_scale=1.0,
        modulation=True,
    ):
        super().__init__()
        
        self.channels = channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.offset_scale = offset_scale
        self.modulation = modulation
        
        assert channels % n_groups == 0, f"channels ({channels}) must be divisible by n_groups ({n_groups})"
        self.channels_per_group = channels // n_groups
        self.n_offset_points = kernel_size * kernel_size
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=n_groups),
            nn.GELU(),
            nn.Conv2d(channels, n_groups * 2 * self.n_offset_points, 
                     kernel_size=3, padding=1, bias=True)
        )
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)
        if modulation:
            self.modulation_conv = nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels // 4, n_groups * self.n_offset_points, 
                         kernel_size=3, padding=1, bias=True)
            )
            nn.init.constant_(self.modulation_conv[-1].weight, 0)
            nn.init.constant_(self.modulation_conv[-1].bias, 0)
        self.register_buffer('base_offset', self._get_base_offset())
        
    def _get_base_offset(self):
        k = self.kernel_size
        range_vals = torch.arange(-(k-1)//2, (k-1)//2+1)
        y, x = torch.meshgrid(range_vals, range_vals, indexing='ij')
        base_offset = torch.stack([x, y], dim=0).float()  # (2, k, k)
        base_offset = base_offset.reshape(2, -1)  # (2, k*k)
        return base_offset
    
    def forward(self, x, patch_resolution):
        B, N, C = x.shape
        H, W = patch_resolution
        assert N == H * W, f"N ({N}) must equal H*W ({H*W})"
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        offset = self.offset_conv(x_2d) * self.offset_scale
        offset = offset.reshape(B, self.n_groups, 2, self.n_offset_points, H, W)
        if self.modulation:
            modulation = torch.sigmoid(self.modulation_conv(x_2d))
            modulation = modulation.reshape(B, self.n_groups, self.n_offset_points, H, W)
        else:
            modulation = None
        output_groups = []
        for g in range(self.n_groups):
            start_c = g * self.channels_per_group
            end_c = (g + 1) * self.channels_per_group
            x_g = x_2d[:, start_c:end_c, :, :] 
            offset_g = offset[:, g, :, :, :, :]
            shifted_g = self._deformable_sample(
                x_g, offset_g, modulation[:, g] if modulation is not None else None
            )
            output_groups.append(shifted_g)
        output = torch.cat(output_groups, dim=1)
        output = output.reshape(B, C, N).transpose(1, 2)
        
        return output
    
    def _deformable_sample(self, x, offset, modulation=None):
        B, C, H, W = x.shape
        n_points = offset.shape[2]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        y_grid = y_grid.reshape(1, 1, H, W)
        x_grid = x_grid.reshape(1, 1, H, W)
        base = self.base_offset.reshape(1, 2, n_points, 1, 1)
        sampling_x = x_grid + base[:, 0] + offset[:, 0]
        sampling_y = y_grid + base[:, 1] + offset[:, 1]
        sampling_x = 2.0 * sampling_x / (W - 1) - 1.0
        sampling_y = 2.0 * sampling_y / (H - 1) - 1.0
        sampling_x = torch.clamp(sampling_x, -1, 1)
        sampling_y = torch.clamp(sampling_y, -1, 1)
        sampled_features = []
        for i in range(n_points):
            grid = torch.stack([sampling_x[:, i], sampling_y[:, i]], dim=-1)  # (B, H, W, 2)
            sampled = F.grid_sample(
                x, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            if modulation is not None:
                mod_weight = modulation[:, i:i+1, :, :]
                sampled = sampled * mod_weight
            
            sampled_features.append(sampled)
        output = torch.stack(sampled_features, dim=0).mean(dim=0)
        
        return output


def da_shift(x, channels, n_groups=4, patch_resolution=None, 
             kernel_size=3, offset_scale=1.0, modulation=True, shift_module=None):
    if shift_module is None:
        shift_module = DeformableAdaptiveShift(
            channels=channels,
            n_groups=n_groups,
            kernel_size=kernel_size,
            offset_scale=offset_scale,
            modulation=modulation
        ).to(x.device)
    
    return shift_module(x, patch_resolution)

