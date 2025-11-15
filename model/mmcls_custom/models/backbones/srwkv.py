import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
from .scan_scan_inv import vertical_forward_scan, vertical_forward_scan_inv, vertical_backward_scan, vertical_backward_scan_inv
from .scan_scan_inv import horizontal_forward_scan, horizontal_forward_scan_inv, horizontal_backward_scan, horizontal_backward_scan_inv


T_MAX = 16384
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
wkv_cuda = load(
    name="wkv", 
    sources=[
        os.path.join(current_dir, 'cuda', 'wkv_op.cpp'), 
        os.path.join(current_dir, 'cuda', 'wkv_cuda.cu')
    ],
    verbose=True, 
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}']
)


class MVCShift(nn.Module):
    def __init__(self, dim, compress_ratio=1, kernel_sizes=(3, 3, 3), dilations=(1, 2, 3)):
        super(MVCShift, self).__init__()
        assert len(kernel_sizes) == len(dilations) == 3, "Kernel sizes and dilations should have 3 values each."
        embed_dim = int(dim * compress_ratio)
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=embed_dim, kernel_size=1, bias=False
        )
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=ks, padding=ks//2 * d, dilation=d,
                      groups=embed_dim, bias=False) for ks, d in zip(kernel_sizes, dilations)
        ])
        self.projs = nn.ModuleList([
            nn.Conv2d(embed_dim, dim, kernel_size=1, bias=False) for _ in range(3)
        ])

    def forward(self, x):
        xx = self.conv1x1(x)
        outs = [self.projs[i](self.dw_convs[i](xx)) for i in range(3)]
        out = x + sum(outs)
        return out


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )
        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training is False and self.repram_flag is True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training is False and self.repram_flag is False:
            out = self.conv5x5_reparam(x)

        return out


def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    if resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
    input = input
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        if T > T_MAX:
            print(f"❌ ERROR: T={T} > T_MAX={T_MAX}")
            print(f"  B={B}, C={C}")
            print(f"  k.shape={k.shape}")
        assert T <= T_MAX, f"T={T} exceeds T_MAX={T_MAX}"
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class SpatialInteractionMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='mvc_shift',
                 channel_gamma=1/4, shift_pixel=1, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        with torch.no_grad():
            ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))

            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            if shift_mode == 'mvc_shift':
                self.shift_func = MVCShift(n_embd)
            elif shift_mode == 'omni_shift':
                self.shift_func = OmniShift(n_embd)
            elif shift_mode == 'da_shift':
                from .da_shift import DeformableAdaptiveShift
                self.shift_func = DeformableAdaptiveShift(
                    channels=n_embd,
                    n_groups=4,
                    kernel_size=3,
                    offset_scale=1.0,
                    modulation=True
                )
            else:
                self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

    
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        
     
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1

    def jit_func(self, x, resolution):
        B, T, C = x.size()
        if self.shift_pixel > 0:
            if self.shift_mode in ['mvc_shift', 'omni_shift']:
                H, W = resolution
                x_2d = x.transpose(1, 2).reshape(B, C, H, W)
                xx_2d = self.shift_func(x_2d)
                xx = xx_2d.flatten(2).transpose(1, 2)
            elif self.shift_mode == 'da_shift':
                xx = self.shift_func(x, resolution)
            else:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # 生成k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class SpectralMixer(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='mvc_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            if shift_mode == 'mvc_shift':
                self.shift_func = MVCShift(n_embd)
            elif shift_mode == 'omni_shift':
                self.shift_func = OmniShift(n_embd)
            elif shift_mode == 'da_shift':
                from .da_shift import DeformableAdaptiveShift
                self.shift_func = DeformableAdaptiveShift(
                    channels=n_embd,
                    n_groups=4,
                    kernel_size=3,
                    offset_scale=1.0,
                    modulation=True
                )
            else:
                self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        self.value.scale_init = 0
        self.receptance.scale_init = 0
        self.key.scale_init = 1

    def forward(self, x, resolution=None):
        if self.shift_pixel > 0:
            if self.shift_mode in ['mvc_shift', 'omni_shift']:
                B, T, C = x.size()
                H, W = resolution
                x_2d = x.transpose(1, 2).reshape(B, C, H, W)
                xx_2d = self.shift_func(x_2d)
                xx = xx_2d.flatten(2).transpose(1, 2)
            elif self.shift_mode == 'da_shift':

                xx = self.shift_func(x, resolution)
            else:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

       
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

       
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class ShapeGuidedRWKVBranch(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, key_norm=True, init_mode='fancy'):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        
        
        with torch.no_grad():
            ratio_0_to_1 = (self.layer_id / max(self.n_layer - 1, 1))
            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / max(self.n_embd - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1
    
    def forward(self, x_seq, scan_indices):
        B, L, C = x_seq.shape
        
        k = self.key(x_seq)      
        v = self.value(x_seq)    
        r = self.receptance(x_seq)  
        sr = torch.sigmoid(r)
        
      
        k_scanned = k[:, scan_indices, :]  
        v_scanned = v[:, scan_indices, :]  
        
       
        rwkv = RUN_CUDA(B, L, C, 
                       self.spatial_decay / L, 
                       self.spatial_first / L, 
                       k_scanned, v_scanned) 
        
       
        inverse_indices = torch.argsort(scan_indices)
        rwkv_restored = rwkv[:, inverse_indices, :]  
        
        
        if self.key_norm is not None:
            rwkv_restored = self.key_norm(rwkv_restored)
        rwkv_restored = sr * rwkv_restored
        out_seq = self.output(rwkv_restored)  
        
        return out_seq


class CrossMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_proj = nn.Conv2d(4, 4, kernel_size=1, bias=True)

    def forward(self, grids):
        assert isinstance(grids, (list, tuple)) and len(grids) == 4
        B, C, H, W = grids[0].shape
        mean_stack = torch.cat([g.mean(dim=1, keepdim=True) for g in grids], dim=1)  # [B, 4, H, W]
        logits = self.weight_proj(mean_stack)
        weights = torch.softmax(logits, dim=1)  # [B, 4, H, W]

        stacked = torch.stack(grids, dim=1)  # [B, 4, C, H, W]
        fused = (stacked * weights.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
        return fused


class ShapeGuidedOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd, n_layer, shift_mode='da_shift', channel_gamma=1/4, 
                 shift_pixel=1, init_mode='fancy', drop_path=0., key_norm=True):
        super().__init__()
        
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            if shift_mode == 'da_shift':
                from .da_shift import DeformableAdaptiveShift
                self.shift_func = DeformableAdaptiveShift(
                    channels=n_embd,
                    n_groups=4,
                    kernel_size=3,
                    offset_scale=1.0,
                    modulation=True
                )
            elif shift_mode == 'mvc_shift':
                self.shift_func = MVCShift(n_embd)
            elif shift_mode == 'omni_shift':
                self.shift_func = OmniShift(n_embd)
            else:
                self.shift_func = eval(shift_mode)
        

        self.scan_types = ['sal_first_h', 'sal_first_v', 'non_sal_first_h', 'non_sal_first_v']
        
      
        self.branches = nn.ModuleList([
            ShapeGuidedRWKVBranch(
                n_embd=self.n_embd,
                n_layer=n_layer,
                layer_id=i,
                key_norm=key_norm,
                init_mode=init_mode
            ) for i in range(4)
        ])
        
      
        self.cross_merge = CrossMerge()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def _process_saliency_mask(self, saliency_mask, B, C, H, W, device):
        if saliency_mask is None:
            return torch.ones(B, 1, H, W, device=device) * 0.5
        
        try:
            
            if saliency_mask.device != device:
                saliency_mask = saliency_mask.to(device)
            
        
            if saliency_mask.dim() == 2:
                saliency_mask = saliency_mask.unsqueeze(0).unsqueeze(0)
            elif saliency_mask.dim() == 3:
                saliency_mask = saliency_mask.unsqueeze(1)
        
          
            if saliency_mask.shape[1] > 1:
                saliency_mask = saliency_mask.mean(dim=1, keepdim=True)
            
          
            if saliency_mask.shape[0] != B:
                saliency_mask = saliency_mask.expand(B, -1, -1, -1)
            
        
            if saliency_mask.shape[2] != H or saliency_mask.shape[3] != W:
                saliency_mask = F.interpolate(
                    saliency_mask, size=(H, W), mode='bilinear', align_corners=False
                )
            
           
            saliency_mask = (saliency_mask >= 0.3).float()
            
            return saliency_mask
            
        except Exception as e:
            print(f"Warning in _process_saliency_mask: {e}")
            return torch.ones(B, 1, H, W, device=device) * 0.5
    
    def _get_saliency_scan_indices(self, sal_mask, scan_type='sal_first_h'):
        H, W = sal_mask.shape[-2:]
        L = H * W
        sal_mask = sal_mask.squeeze() 
        sal_positions = torch.nonzero(sal_mask > 0.5, as_tuple=False) 
        non_sal_positions = torch.nonzero(sal_mask <= 0.5, as_tuple=False)  
        if len(sal_positions) == 0 or len(non_sal_positions) == 0:
            return torch.arange(L, device=sal_mask.device)
        if scan_type == 'sal_first_h':
            sal_idx = sal_positions[:, 0] * W + sal_positions[:, 1]
            sal_idx, _ = torch.sort(sal_idx)
            non_sal_idx = non_sal_positions[:, 0] * W + non_sal_positions[:, 1]
            non_sal_idx, _ = torch.sort(non_sal_idx)
            indices = torch.cat([sal_idx, non_sal_idx])
            
        elif scan_type == 'sal_first_v':
            sal_idx = sal_positions[:, 1] * H + sal_positions[:, 0]
            sal_idx, _ = torch.sort(sal_idx)
            sal_idx = (sal_idx % H) * W + (sal_idx // H)
            non_sal_idx = non_sal_positions[:, 1] * H + non_sal_positions[:, 0]
            non_sal_idx, _ = torch.sort(non_sal_idx)
            non_sal_idx = (non_sal_idx % H) * W + (non_sal_idx // H)
            indices = torch.cat([sal_idx, non_sal_idx])
            
        elif scan_type == 'non_sal_first_h':
            non_sal_idx = non_sal_positions[:, 0] * W + non_sal_positions[:, 1]
            non_sal_idx, _ = torch.sort(non_sal_idx)
            sal_idx = sal_positions[:, 0] * W + sal_positions[:, 1]
            sal_idx, _ = torch.sort(sal_idx)
            indices = torch.cat([non_sal_idx, sal_idx])
            
        else:  
            non_sal_idx = non_sal_positions[:, 1] * H + non_sal_positions[:, 0]
            non_sal_idx, _ = torch.sort(non_sal_idx)
            non_sal_idx = (non_sal_idx % H) * W + (non_sal_idx // H)
            sal_idx = sal_positions[:, 1] * H + sal_positions[:, 0]
            sal_idx, _ = torch.sort(sal_idx)
            sal_idx = (sal_idx % H) * W + (sal_idx // H)
            indices = torch.cat([non_sal_idx, sal_idx])
        
        return indices
    
    def forward(self, x, saliency_mask=None):
        B, C, H, W = x.shape
        L = H * W
        resolution = (H, W)
        x_seq = x.flatten(2).transpose(1, 2)  
        if self.shift_pixel > 0:
            if self.shift_mode == 'da_shift':
                x_shifted = self.shift_func(x_seq, resolution) 
            else:
                x_2d = x_seq.transpose(1, 2).reshape(B, C, H, W)
                x_2d_shifted = self.shift_func(x_2d)
                x_shifted = x_2d_shifted.flatten(2).transpose(1, 2)
        else:
            x_shifted = x_seq
        sal_mask = self._process_saliency_mask(saliency_mask, B, C, H, W, x.device)
        
        outputs = []
        for i, (branch, scan_type) in enumerate(zip(self.branches, self.scan_types)):
            scan_indices = self._get_saliency_scan_indices(sal_mask[0], scan_type) 
            out_group_seq = branch(x_shifted, scan_indices) 
            outputs.append(out_group_seq)

        grids = [o.transpose(1, 2).reshape(B, C, H, W) for o in outputs] 
        fused = self.cross_merge(grids)  
        out = x + self.drop_path(fused)
        return out
