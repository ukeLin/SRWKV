import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import scipy.stats as st
from timm.models.layers import DropPath
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from .mmcls_custom.models.backbones.vrwkv import VRWKV_Hierarchical
from .mmcls_custom.models.backbones.srwkv import ShapeGuidedOrientatedRWKV2D
from .encoder import Encoder_B, Encoder_S, Encoder_T
from .ccm import CCMix


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ChannelFusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ChannelFusionConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.reshape(B, -1, self.scale_factor, self.scale_factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, -1, H * self.scale_factor, W * self.scale_factor)
        return x

class HA(nn.Module):
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = self._gkern(31, 4)
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = nn.Parameter(torch.from_numpy(gaussian_kernel), requires_grad=False)
    
    @staticmethod
    def _gkern(kernlen=16, nsig=3):
        interval = (2*nsig+1.)/kernlen
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return np.float32(kernel)
    
    @staticmethod
    def _min_max_norm(in_):
        max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
        min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
        in_ = in_ - min_
        eps = 1e-3
        norm_factor = torch.clamp(max_ - min_, min=eps)
        return in_.div(norm_factor)
    
    def forward(self, attention):
        attention = torch.clamp(attention, min=0, max=1)
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = self._min_max_norm(soft_attention)
        soft_attention = torch.nan_to_num(soft_attention, nan=0.0, posinf=1.0, neginf=0.0)
        x = torch.max(soft_attention, attention)
        return x


class SaliencyPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pred_saliency = nn.Conv2d(in_channels[-1], 1, kernel_size=1, bias=False)
        self.ha = HA()
    
    def forward(self, features):
        feat_last = features[-1]
        saliency = self.pred_saliency(F.interpolate(feat_last, scale_factor=16, mode='bilinear', align_corners=False))
        saliency = torch.clamp(saliency, min=-10, max=10)
        guide_saliency = torch.sigmoid(saliency)
        guide_saliency = self.ha(guide_saliency)
        guide_saliency = torch.clamp(guide_saliency, min=0, max=1)
        
        return saliency, guide_saliency


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None, n_layer=12, scale_factor=2, 
                 shift_mode='da_shift'):
        super().__init__()
        assert in_channels % 4 == 0
        
        if skip_channels is None:
            skip_channels = in_channels
        
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)
        
        self.shape_guided_rwkv = ShapeGuidedOrientatedRWKV2D(
            n_embd=in_channels,
            n_layer=n_layer,
            shift_mode=shift_mode,
            channel_gamma=1,
            shift_pixel=1,
            key_norm=True
        )
        
        self.skip_proj = None
        if skip_channels != in_channels:
            self.skip_proj = nn.Conv2d(skip_channels, in_channels, 1)
        
        self.concat_layer = ChannelFusionConv(2 * in_channels, in_channels)

        self.upsample = PatchExpand(in_channels, out_channels, scale_factor)
        
    def forward(self, x, skip, saliency_mask):
        if saliency_mask.device != x.device:
            saliency_mask = saliency_mask.to(x.device)
    
        x = self.norm1(x)
        x_sg = self.shape_guided_rwkv(x, saliency_mask)
        
        if skip is not None:
            skip_feat = skip
            if self.skip_proj is not None:
                skip_feat = self.skip_proj(skip_feat)
            
            x_cat = torch.cat([x_sg, skip_feat], dim=1)
            x_fused = self.concat_layer(x_cat)
        else:
            x_fused = x_sg
        
        x_fused = self.norm2(x_fused)
        out = self.upsample(x_fused)
        
        return out


class SRWKV(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, img_size=224, encoder_pretrained_path=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dims = [32, 64, 128, 192]
        self.encoder = Encoder_S(embed_dims=self.embed_dims, img_size=img_size)
        
        if encoder_pretrained_path is not None:
            self.load_encoder_pretrained(encoder_pretrained_path)
        

        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], 
                         self.embed_dims[0], img_size//2)

        self.saliency_predictor = SaliencyPredictor(
            in_channels=[self.embed_dims[0], self.embed_dims[1], self.embed_dims[2], self.embed_dims[3]]
        )
        
        self.decoder_stage4 = DecoderBlock(192, 128, skip_channels=192, shift_mode='da_shift', scale_factor=2)
        self.decoder_stage3 = DecoderBlock(128, 64, skip_channels=128, shift_mode='da_shift', scale_factor=2)
        self.decoder_stage2 = DecoderBlock(64, 32, skip_channels=64, shift_mode='da_shift', scale_factor=2)
        self.decoder_stage1 = DecoderBlock(32, 16, skip_channels=32, shift_mode='da_shift', scale_factor=2)
        
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def load_encoder_pretrained(self, pretrained_path):
        print(f"Loading encoder pretrained weights from: {pretrained_path}")
        try:
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            elif 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            
            pretrained_params = sum(p.numel() for p in pretrained_dict.values())
            encoder_params_before = {name: param.clone() for name, param in self.encoder.named_parameters()}
            msg = self.encoder.load_state_dict(pretrained_dict, strict=False)
            
            loaded_params = 0
            loaded_keys = 0
            for name, param in self.encoder.named_parameters():
                if name in pretrained_dict:
                    loaded_params += param.numel()
                    loaded_keys += 1
            
            total_encoder_params = sum(p.numel() for p in self.encoder.parameters())
            
            print(f"✓ Encoder pretrained weights loaded successfully!")
            print(f"  Pretrained file contains: {pretrained_params:,} parameters")
            print(f"  Encoder total parameters: {total_encoder_params:,}")
            print(f"  Loaded parameters: {loaded_params:,} ({loaded_params/total_encoder_params*100:.1f}%)")
            print(f"  Loaded keys: {loaded_keys}")
            print(f"  Missing keys: {len(msg.missing_keys)}")
            print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
            
            if msg.missing_keys:
                print(f"\n  Missing keys (first 10): {msg.missing_keys[:10]}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys (first 10): {msg.unexpected_keys[:10]}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not load encoder pretrained weights: {e}")
            print(f"  Continuing with randomly initialized encoder...")
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3, :, :]
        
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        
        for blk in self.encoder.stage4:
            x = blk(x)
        enc4 = x
        
        enc3_fused, enc2_fused, enc1_fused = self.ccm([enc3, enc2, enc1])
        saliency, guide_saliency = self.saliency_predictor([enc1_fused, enc2_fused, enc3_fused, enc4])
        
        def soft_threshold(x, threshold=0.3, sharpness=10.0):
            return torch.sigmoid(sharpness * (x - threshold))
        
        H4, W4 = enc4.shape[2], enc4.shape[3]
        saliency_mask4 = F.interpolate(guide_saliency, size=(H4, W4), mode='bilinear', align_corners=False)
        saliency_mask4 = soft_threshold(saliency_mask4, threshold=0.3, sharpness=10.0)
        
        H3, W3 = enc3_fused.shape[2], enc3_fused.shape[3]
        saliency_mask3 = F.interpolate(guide_saliency, size=(H3, W3), mode='bilinear', align_corners=False)
        saliency_mask3 = soft_threshold(saliency_mask3, threshold=0.3, sharpness=10.0)
        
        H2, W2 = enc2_fused.shape[2], enc2_fused.shape[3]
        saliency_mask2 = F.interpolate(guide_saliency, size=(H2, W2), mode='bilinear', align_corners=False)
        saliency_mask2 = soft_threshold(saliency_mask2, threshold=0.3, sharpness=10.0)
        
        H1, W1 = enc1_fused.shape[2], enc1_fused.shape[3]
        saliency_mask1 = F.interpolate(guide_saliency, size=(H1, W1), mode='bilinear', align_corners=False)
        saliency_mask1 = soft_threshold(saliency_mask1, threshold=0.3, sharpness=10.0)
        
        dec4 = self.decoder_stage4(enc4, None, saliency_mask4)
        dec3 = self.decoder_stage3(dec4, enc3_fused, saliency_mask3)
        dec2 = self.decoder_stage2(dec3, enc2_fused, saliency_mask2)
        dec1 = self.decoder_stage1(dec2, enc1_fused, saliency_mask1)
        
        out = self.seg_head(dec1)
        
        return out
