# from thop import profile

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import Tuple, Optional, List

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def computer(tensor):
    def get_stats(t):
        return {
            "max": torch.max(t).item(),
            "min": torch.min(t).item(),
            "mean": torch.mean(t).item(),
            "var": torch.var(t).item()
        }

    # 分别计算两个张量的指标
    stats1 = get_stats(tensor)
    return stats1
    # print(f"Tensor Stats: max={stats1['max']:.4f}, min={stats1['min']:.4f}, mean={stats1['mean']:.4f}, var={stats1['var']:.4f}")
class StructureAgnosticMotionAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        
        # 1. 内部映射 (Q, K, V) - 相当于 LoRA A
        self.to_q = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.to_k = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.to_v = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        
        # Norm 层
        self.norm_q = nn.GroupNorm(32, in_channels)
        self.norm_kv = nn.GroupNorm(32, in_channels)
        
        # 2. 输出映射 - 相当于 LoRA B (最后一层)
        self.out_proj = nn.Conv3d(in_channels, in_channels, 1)
        
        # 执行初始化
        self.reset_parameters()

    def reset_parameters(self):
        # A. 内部层 Xavier
        torch.nn.init.xavier_normal_(self.to_q.weight)
        torch.nn.init.xavier_normal_(self.to_k.weight)
        torch.nn.init.xavier_normal_(self.to_v.weight)
        
        # B. 输入 Norm 层初始化 (标准)
        for m in [self.norm_q, self.norm_kv]:
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
            
        # C. 输出层 Zero Init (关键)
        # 确保 Conv3d 的权重和偏置都为 0
        torch.nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            torch.nn.init.zeros_(self.out_proj.bias)

    def forward(self, feat_1d, feat_2d):
        """
        feat_1d (Q): (B, C, F, H, W) - 运动向量，实际上通常H,W维度是广播复制的，或者包含全局信息
        feat_2d (K,V): (B, C, F, H, W) - 大人的骨骼图
        """
        b, c, f, h, w = feat_2d.shape
        fead_pad, feat_1d = torch.split(feat_1d, [1, f], dim=2)
        # 1. 预处理
        q_in = self.norm_q(feat_1d)
        kv_in = self.norm_kv(feat_2d)
        q = self.to_q(q_in) # (B, C, F, H, W)
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)
        
        head_dim = c // self.num_heads
        
        # -------------------------------------------------------------------------
        # 关键步骤：构建去空间化的 Key/Value (Bag of Motion Features)
        # -------------------------------------------------------------------------
        # 我们不希望 Q 在 (x,y) 位置只能看 K 在 (x,y) 的位置。
        # 我们希望 Q 在任何位置都能看到 K 的所有位置，并根据语义聚合。
        
        # 策略：保持时间维度 F 对齐（帧对帧），但打平空间维度 H, W
        # Q: (B, Heads, F, H*W, Head_Dim) 
        # K, V: (B, Heads, F, H*W, Head_Dim)
        
        q = q.view(b, self.num_heads, head_dim, f, feat_1d.shape[3]*feat_1d.shape[4]).permute(0, 1, 3, 4, 2) 
        # -> (B, Heads, F, HW, Dim)
        
        k = k.view(b, self.num_heads, head_dim, f, h*w).permute(0, 1, 3, 4, 2)
        # -> (B, Heads, F, HW, Dim)
        
        v = v.view(b, self.num_heads, head_dim, f, h*w).permute(0, 1, 3, 4, 2)
        # -> (B, Heads, F, HW, Dim)

        # -------------------------------------------------------------------------
        # Attention 计算: Q (Motion) 查询 K (Structure)
        # -------------------------------------------------------------------------
        # 为了彻底解耦空间，我们让 Q 的每个空间点都能 attend 到 K 的所有空间点。
        # 这里的 seq_len = HW。
        
        # attn_score: (B, Heads, F, HW_q, HW_k)
        # 这是一个巨大的 Attention Map，表示：
        # "1D Motion 在位置 i (虽然它是广播的) 与 2D Pose 在位置 j 的特征有多像？"
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 聚合特征
        # out: (B, Heads, F, HW_q, Dim)
        out = attn @ v
        
        # 恢复形状
        out = out.permute(0, 1, 4, 2, 3).contiguous() # (B, Heads, Dim, F, HW)
        out = out.view(b, c, f, feat_1d.shape[3], feat_1d.shape[4])
        # -------------------------------------------------------------------------
        # 残差连接的特殊处理
        # -------------------------------------------------------------------------
        # 这里非常重要：
        # 千万不要做 out + feat_2d (大人的2D图)。因为 feat_2d 带有大人的体型。
        # 应该做 out + feat_1d (如果你认为1D包含了足够的信息) 
        # 或者直接输出 out (纯粹的重组特征)。
        # 建议加回 feat_1d，因为它是位置不敏感的基准。
        out = self.out_proj(out) + feat_1d
        return torch.cat([fead_pad, out], dim=2)

class StructureAgnosticMotionAttention1tox(StructureAgnosticMotionAttention):
    
    def qkv(self, q, k, v, shape, feat_1d):
        b, c, f, h, w = shape
        head_dim = c // self.num_heads
        
        # -------------------------------------------------------------------------
        # 关键步骤：构建去空间化的 Key/Value (Bag of Motion Features)
        # -------------------------------------------------------------------------
        # 我们不希望 Q 在 (x,y) 位置只能看 K 在 (x,y) 的位置。
        # 我们希望 Q 在任何位置都能看到 K 的所有位置，并根据语义聚合。
        
        # 策略：保持时间维度 F 对齐（帧对帧），但打平空间维度 H, W
        # Q: (B, Heads, F, H*W, Head_Dim) 
        # K, V: (B, Heads, F, H*W, Head_Dim)
        
        q = q.view(b, self.num_heads, head_dim, f, feat_1d.shape[3]*feat_1d.shape[4]).permute(0, 1, 3, 4, 2) 
        # -> (B, Heads, F, HW, Dim)
        
        k = k.view(b, self.num_heads, head_dim, f, h*w).permute(0, 1, 3, 4, 2)
        # -> (B, Heads, F, HW, Dim)
        
        v = v.view(b, self.num_heads, head_dim, f, h*w).permute(0, 1, 3, 4, 2)
        # -> (B, Heads, F, HW, Dim)

        # -------------------------------------------------------------------------
        # Attention 计算: Q (Motion) 查询 K (Structure)
        # -------------------------------------------------------------------------
        # 为了彻底解耦空间，我们让 Q 的每个空间点都能 attend 到 K 的所有空间点。
        # 这里的 seq_len = HW。
        
        # attn_score: (B, Heads, F, HW_q, HW_k)
        # 这是一个巨大的 Attention Map，表示：
        # "1D Motion 在位置 i (虽然它是广播的) 与 2D Pose 在位置 j 的特征有多像？"
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 聚合特征
        # out: (B, Heads, F, HW_q, Dim)
        out = attn @ v
        
        # 恢复形状
        out = out.permute(0, 1, 4, 2, 3).contiguous() # (B, Heads, Dim, F, HW)
        out = out.view(b, c, f, feat_1d.shape[3], feat_1d.shape[4])
        
        return out
        
    def forward(self, feat_1d, feat_2d):
        """
        feat_1d (Q): (B, C, F, H, W) - 运动向量，实际上通常H,W维度是广播复制的，或者包含全局信息
        feat_2d (K,V): (B, R, C, F, H, W) - 大人的骨骼图，帧数可以不同（比如2倍）
        """
        b, r, c, f, h, w = feat_2d.shape
        fead_pad, feat_1d = torch.split(feat_1d, [1, f], dim=2)
        # 1. 预处理
        q_in = self.norm_q(feat_1d)
        outlist = []
        for _ in range(r):
            kv_in = self.norm_kv(feat_2d[:, _])
            k = self.to_k(kv_in)
            v = self.to_v(kv_in)
            out = self.qkv(q_in, k, v, feat_2d[:, _].shape, feat_1d)
            outlist.append(out)
        out = torch.mean(torch.stack(outlist, dim=0), dim=0)
        out = self.out_proj(out) + feat_1d
        return torch.cat([fead_pad, out], dim=2)
        

class FrameWiseSpatialCrossAttn(nn.Module):
    """
    针对 Rescale 和空间不对齐优化的 Cross-Attention 模块
    Q: Driven Pose (确定输出的拓扑结构)
    K, V: Ref Pose (提供原始外观/姿态特征)
    """
    def __init__(
        self,
        dim: int = 1536,
        heads: int = 12,
        height: int = 64,
        width: int = 64,
        residual_inject: bool = True,
        dropout: float = 0.0,
        return_attn: bool = False
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.residual_inject = residual_inject
        self.return_attn = return_attn

        # --- 提前计算并注册位置编码 ---
        # 注册为 buffer，它不属于 Parameter 但会随模型移动设备和转换类型 (float32 -> bfloat16)
        pos_embed = self._init_2d_pos_embed(dim, height, width)
        self.register_buffer("pos_embed", pos_embed) # (1, H*W, C)

        # 层归一化与线性投影
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def _init_2d_pos_embed(self, dim, h, w):
        """生成 2D Sine-Cosine 位置编码 (内部初始化调用)"""
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        
        pos_dim = dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (10000**omega)
        
        out_h = torch.einsum('m,d->md', grid_h.flatten(), omega)
        out_w = torch.einsum('m,d->md', grid_w.flatten(), omega)
        
        pos_emb = torch.cat([
            torch.sin(out_h), torch.cos(out_h),
            torch.sin(out_w), torch.cos(out_w)
        ], dim=1) # (H*W, C)
        return pos_emb.unsqueeze(0) # (1, N, C)

    def forward(self, ref_pose: torch.Tensor, driven_pose: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        ref_pose:    (B, C, 1, H, W)
        driven_pose: (B, C, T, H, W)
        """
        B, C, _, H, W = ref_pose.shape
        _, _, T, _, _ = driven_pose.shape
        # 获取当前运行的 dtype (解决 float != bfloat16 报错)
        target_dtype = ref_pose.dtype

        # 1. 处理 K, V (来自 Reference)
        # (B, C, 1, H, W) -> (B, N, C)
        kv_tokens = ref_pose[:, :, 0].permute(0, 2, 3, 1).reshape(B, H * W, C)
        kv_tokens = self.norm_kv(kv_tokens)
        
        # K 建议也加入位置编码，增强对应的空间检索能力
        # k = self.k_proj(kv_tokens)
        k = self.k_proj(kv_tokens + self.pos_embed.to(target_dtype))
        k = k.view(B, H * W, self.heads, self.head_dim).transpose(1, 2) # (B, heads, N, dh)
        
        v = self.v_proj(kv_tokens).view(B, H * W, self.heads, self.head_dim).transpose(1, 2)

        # 2. 处理 Q (来自 Driven Sequence)
        # (B, C, T, H, W) -> (B*T, N, C)
        q_tokens = driven_pose.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        q_tokens = self.norm_q(q_tokens)
        
        # 核心：将提前算好的位置编码转为当前 dtype 并相加
        q_tokens = q_tokens + self.pos_embed.to(target_dtype)
        
        q = self.q_proj(q_tokens).view(B * T, H * W, self.heads, self.head_dim).transpose(1, 2)
        # import pudb; pudb.set_trace()
        # 3. 准备 Cross-Attention (K, V 广播到 B*T 维度)
        # 使用 expand 代替 repeat_interleave 更省显存
        k = k.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, self.heads, H * W, self.head_dim)
        v = v.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, self.heads, H * W, self.head_dim)

        # 4. Attention 计算: (B*T, heads, N_q, N_kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # np_attn = attn.detach().cpu().float().numpy()
        # np.save('attn.npy', np_attn)
        # 5. 输出投影与还原
        out = torch.matmul(attn, v) # (B*T, heads, N, dh)
        out = out.transpose(1, 2).contiguous().reshape(B * T, H * W, C)
        out = self.out_drop(self.out_proj(out))

        # (B*T, N, C) -> (B, C, T, H, W)
        aligned_seq = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        if self.residual_inject:
            # 残差连接在驱动姿态上
            aligned_seq = aligned_seq + driven_pose

        return aligned_seq   ##, attn if self.return_attn else None

class WanCustom(torch.nn.Module):
    def __init__(self, in_channels=32, out_channels=16, patch_size=[1, 2, 2]):
        super().__init__()
        self.pose_patch_embedding = nn.Conv3d(16, 1536, kernel_size=patch_size, stride=patch_size)
        
        self.motion_proj = nn.Linear(512, 32)
        self.norm_out = nn.LayerNorm(32)
        self.motion_cond_projs = nn.ModuleList()
        for d in range(30 // 2 - 1):
            l = nn.Linear(32, 1536)
            zero_module(l)
            self.motion_cond_projs.append(l)  
        
        torch.nn.init.xavier_normal_(self.pose_patch_embedding.weight)
        torch.nn.init.zeros_(self.pose_patch_embedding.bias)
        # self.align_model = FrameWiseSpatialCrossAttn(dim=1536, heads=12, height=52, width=30, residual_inject=True, return_attn=False)
        self.align_model = FrameWiseSpatialCrossAttn(dim=1536, heads=12, height=80, width=45, residual_inject=True, return_attn=False)

        self.attn = StructureAgnosticMotionAttention(in_channels=1536)
        # self.attn = StructureAgnosticMotionAttention1tox(in_channels=1536)
        
    def patchify(self, x: torch.Tensor):
        x = self.pose_patch_embedding(x)
        # x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x 
    
    def motion_patchify(self, x: torch.Tensor):
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x
    
# class WanCustomMotion(nn.Module):
#     def __init__(self): 
#         super().__init__()

#         self.motion_proj = nn.Linear(512, 32)
#         self.norm_out = nn.LayerNorm(32)
#         self.motion_cond_projs = nn.ModuleList()
#         for d in range(30 // 2 - 1):
#             l = nn.Linear(32, 1536)
#             zero_module(l)
#             self.motion_cond_projs.append(l)  
            
#     def forward(self, motion_latents):
#         motion_latents = motion_latents[:,:,:, None, None]
#         motion_latents = rearrange(motion_latents, 'b f c h w -> b f h w c').contiguous()
#         motion_latents = self.motion_proj(motion_latents)
#         motion_latents = self.norm_out(motion_latents)
#         motion_latents = torch.concat([motion_cond_proj(motion_latents) for motion_cond_proj in self.motion_cond_projs], 0)
#         motion_pad = torch.zeros_like(motion_latents[:, :1], device=motion_latents.device, dtype=motion_latents.dtype)  # b, 1, c, h, w
#         motion_latents = torch.cat([motion_pad, motion_latents], dim=1)
#         motion_latents = motion_latents.reshape(motion_latents.shape[0], motion_latents.shape[0] // motion_latents.shape[0], -1, *motion_latents.shape[2:])  # b, f, c, h, w)
#         return motion_latents    

# if __name__ == "__main__":
#     motion_latents = torch.randn(1,11,512)
#     model = WanCustomMotion()
#     flops, params = profile(model, inputs=(motion_latents,))
#     print(f"FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.2f} M")