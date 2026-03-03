import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import cv2
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.nn.functional as F
from mmpose.apis import init_model

from posetimation import get_cfg, update_config

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, context, attn_mask=None):

        attn_output, attn_weight = self.mha(query, context, context, attn_mask=attn_mask)
        
        # Add residual connection and layer norm
        attn_output = self.norm(query + attn_output)
        
        return attn_output, attn_weight   

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.mha(x, x, x)

        # Add residual connection and layer norm
        attn_output = self.norm(x + attn_output)

        return attn_output

class FFNLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        h = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x + h)
        return x

class TAR_ViTPose(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4):
        super(TAR_ViTPose, self).__init__()
        
        self.device = device

        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        self.backbone = self.model.backbone
        
        self.heatmap_head = nn.Sequential(
            self.model.head.deconv_layers,
            self.model.head.final_layer
        )

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_frames = cfg.WINDOWS_SIZE

        self.f_w = self.heatmap_size[0] // 4
        self.f_h = self.heatmap_size[1] // 4
        
        self.is_train = True if phase == 'train' else False

        self.query_feat = nn.Embedding(self.num_joints, self.embed_dim)
        self.query_pe = nn.Embedding(self.num_joints, self.embed_dim)
        self.mask_threshold = cfg.MODEL.MASK_THRESHOLD
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.use_mask = cfg.MODEL.USE_MASK

        self.masked_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.masked_attention_layers.append(CrossAttention(self.embed_dim, self.num_heads))
            self.self_attention_layers.append(SelfAttention(self.embed_dim, self.num_heads))
            self.ffn_layers.append(FFNLayer(self.embed_dim, hidden_features=self.embed_dim * 2, out_features=self.embed_dim))

        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)

        # Print learning parameters
        print(f"TAR_ViTPose learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")

        print(f"TAR_ViTPose parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

        print(f"TAR_ViTPose backbone parameters: {round(self.count_backbone_parameters() / 1e6, 1)} M\n\n")

    def generate_attn_mask(self, heatmap, output_size=(24, 18), threshold=0.2):

        B, J, _, _ = heatmap.shape

        min_val = heatmap.amin(dim=(2, 3), keepdim=True)  
        max_val = heatmap.amax(dim=(2, 3), keepdim=True) 
        heatmap = (heatmap - min_val) / (max_val - min_val + 1e-6) 

        # Downsample the heatmap
        heatmap_downsampled = F.interpolate(heatmap, size=output_size, mode='bilinear', align_corners=False) 
        mask = (heatmap_downsampled < threshold).bool()
        mask = mask.view(B // self.num_frames, self.num_frames, J, output_size[0], output_size[1])

        attn_mask = mask.flatten(3).unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()

        return attn_mask
    
    
    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone
        x = self.backbone(x)[0]
        x = x.view(batch_size*num_frames, self.embed_dim, self.f_h, self.f_w)

        init_heatmap = self.heatmap_head(x)

        x = x.view(batch_size, num_frames, self.embed_dim, self.f_h, self.f_w)

        attn_mask = self.generate_attn_mask(init_heatmap, output_size=(self.f_h, self.f_w), threshold=self.mask_threshold)

        # (B, J, C)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pe = self.query_pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        output = query_feat + query_pe

        x = x.flatten(3).reshape(batch_size, num_frames, self.f_h*self.f_w, self.embed_dim)
        center_frame_idx = num_frames // 2
        center_frame = x[:, center_frame_idx]

        x = x.flatten(1, 2)
        attn_mask = attn_mask.view(-1, self.num_joints, num_frames, self.f_h*self.f_w).flatten(2)
        # attn_maps = []

        for i in range(self.num_layers):
            
            # Cross-Attention
            if self.use_mask:
                output, attn_weight = self.masked_attention_layers[i](output, x, attn_mask=attn_mask)
            else:
                output, attn_weight = self.masked_attention_layers[i](output, x)

            # attn_maps.append(attn_weight)
            
            # Self-Attention
            output = self.self_attention_layers[i](output)

            # Feed-Forward Network
            output = self.ffn_layers[i](output)

        center_frame, _ = self.cross_attention(center_frame, output)

        center_frame = center_frame.view(batch_size, self.embed_dim, self.f_h, self.f_w)
        x = self.heatmap_head(center_frame)
        
        return x

    def count_backbone_parameters(self):
        return sum(p.numel() for p in self.backbone.parameters())

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def set_phase(self, phase):
        self.phase = phase
        self.is_train = True if phase == TRAIN_PHASE else False

    def get_phase(self):
        return self.phase

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            norm_factor=np.ones((N, 2), dtype=np.float32))

        return avg_acc



