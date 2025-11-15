import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAdaptiveShift(nn.Module):
    def __init__(
        self, 
        channels,
        n_groups=2,
        kernel_size=3,
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
        
        # Number of offset points = kernel_size^2
        self.n_offset_points = kernel_size * kernel_size
        
        # Offset prediction network: predicts (x, y) offsets for each point
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=n_groups),
            nn.GELU(),
            nn.Conv2d(channels, n_groups * 2 * self.n_offset_points, 
                     kernel_size=3, padding=1, bias=True)
        )
        
        # Initialize offset prediction to zero (start with regular grid)
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)
        
        # Modulation network: predicts importance weights for each offset point
        if modulation:
            self.modulation_conv = nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels // 4, n_groups * self.n_offset_points, 
                         kernel_size=3, padding=1, bias=True)
            )
            nn.init.constant_(self.modulation_conv[-1].weight, 0)
            nn.init.constant_(self.modulation_conv[-1].bias, 0)
        
        # Base grid positions (regular grid offsets)
        self.register_buffer('base_offset', self._get_base_offset())
        
    def _get_base_offset(self):
        """Generate base offset grid (regular grid centered at origin)"""
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
        
        # Reshape to (B, C, H, W) for 2D operations
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Predict offsets: (B, n_groups * 2 * n_offset_points, H, W)
        offset = self.offset_conv(x_2d) * self.offset_scale
        offset = offset.reshape(B, self.n_groups, 2, self.n_offset_points, H, W)
        
        # Predict modulation weights if enabled
        if self.modulation:
            modulation = torch.sigmoid(self.modulation_conv(x_2d))
            modulation = modulation.reshape(B, self.n_groups, self.n_offset_points, H, W)
        else:
            modulation = None
        
        # Apply deformable shift for each group
        output_groups = []
        for g in range(self.n_groups):
            # Extract channels for this group
            start_c = g * self.channels_per_group
            end_c = (g + 1) * self.channels_per_group
            x_g = x_2d[:, start_c:end_c, :, :]  # (B, C_per_group, H, W)
            
            # Extract offsets for this group
            offset_g = offset[:, g, :, :, :, :]  # (B, 2, n_points, H, W)
            
            # Apply deformable sampling
            shifted_g = self._deformable_sample(
                x_g, offset_g, modulation[:, g] if modulation is not None else None
            )
            output_groups.append(shifted_g)
        
        # Concatenate all groups
        output = torch.cat(output_groups, dim=1)  # (B, C, H, W)
        
        # Reshape back to (B, N, C)
        output = output.reshape(B, C, N).transpose(1, 2)
        
        return output
    
    def _deformable_sample(self, x, offset, modulation=None):
        B, C, H, W = x.shape
        n_points = offset.shape[2]
        
        # Create sampling grid
        # Base grid: (H, W) positions
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        # (1, 1, H, W)
        y_grid = y_grid.reshape(1, 1, H, W)
        x_grid = x_grid.reshape(1, 1, H, W)
        
        # Add base offsets and predicted offsets
        # base_offset: (2, n_points) -> (1, 2, n_points, 1, 1)
        base = self.base_offset.reshape(1, 2, n_points, 1, 1)
        
        # sampling_x: (B, n_points, H, W)
        sampling_x = x_grid + base[:, 0] + offset[:, 0]
        sampling_y = y_grid + base[:, 1] + offset[:, 1]
        
        # Normalize to [-1, 1] for grid_sample
        sampling_x = 2.0 * sampling_x / (W - 1) - 1.0
        sampling_y = 2.0 * sampling_y / (H - 1) - 1.0
        
        # Clamp to valid range
        sampling_x = torch.clamp(sampling_x, -1, 1)
        sampling_y = torch.clamp(sampling_y, -1, 1)
        
        # Sample features at offset positions
        # For each offset point, sample the feature
        sampled_features = []
        for i in range(n_points):
            grid = torch.stack([sampling_x[:, i], sampling_y[:, i]], dim=-1)  # (B, H, W, 2)
            sampled = F.grid_sample(
                x, grid, mode='bilinear', padding_mode='border', align_corners=True
            )  # (B, C, H, W)
            
            # Apply modulation if available
            if modulation is not None:
                mod_weight = modulation[:, i:i+1, :, :]  # (B, 1, H, W)
                sampled = sampled * mod_weight
            
            sampled_features.append(sampled)
        
        # Aggregate: average over all offset points
        output = torch.stack(sampled_features, dim=0).mean(dim=0)  # (B, C, H, W)
        
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


# Example usage and comparison
if __name__ == '__main__':
    # Test DA-Shift
    B, H, W, C = 2, 16, 16, 64
    N = H * W
    
    x = torch.randn(B, N, C).cuda()
    patch_resolution = (H, W)
    
    # Initialize DA-Shift
    da_shift_module = DeformableAdaptiveShift(
        channels=C,
        n_groups=4,
        kernel_size=3,
        offset_scale=1.0,
        modulation=True
    ).cuda()
    
    # Apply DA-Shift
    output = da_shift_module(x, patch_resolution)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"DA-Shift module parameters: {sum(p.numel() for p in da_shift_module.parameters()):,}")

