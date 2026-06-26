import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import Tuple, Optional, List


# class FrameWiseSpatialCrossAttn(nn.Module):
#     """
#     Frame-wise spatial cross-attention:
#       For each t:
#         Q = ref tokens (H*W)
#         K,V = driven tokens at frame t (H*W)
#         out_t = Attn(Q,K,V) -> (B,C,1,H,W)
#       Stack t -> (B,C,T,H,W)

#     Inputs:
#       ref_pose:    (B, C, 1, H, W)
#       driven_pose: (B, C, T, H, W)

#     Returns:
#       aligned_seq: (B, C, T, H, W)
#       attn_maps:   Optional[List[Tensor]] each (B, heads, N, N) if return_attn=True
#     """
#     def __init__(
#         self,
#         dim: int = 1536,
#         heads: int = 12,
#         use_delta: bool = True,     # use driven_t - driven_0 as K/V
#         residual_inject: bool = True,  # aligned_t = ref + out_t
#         dropout: float = 0.0,
#         return_attn: bool = False
#     ):
#         super().__init__()
#         assert dim % heads == 0, "dim must be divisible by heads"
#         self.dim = dim
#         self.heads = heads
#         self.head_dim = dim // heads
#         self.scale = 1.0 / math.sqrt(self.head_dim)

#         self.use_delta = use_delta
#         self.residual_inject = residual_inject
#         self.return_attn = return_attn

#         # LayerNorm over channel dim after flattening to (B, N, C)
#         self.norm_q = nn.LayerNorm(dim)
#         self.norm_kv = nn.LayerNorm(dim)

#         self.q_proj = nn.Linear(dim, dim, bias=False)
#         self.k_proj = nn.Linear(dim, dim, bias=False)
#         self.v_proj = nn.Linear(dim, dim, bias=False)
#         self.out_proj = nn.Linear(dim, dim, bias=False)

#         self.attn_drop = nn.Dropout(dropout)
#         self.out_drop = nn.Dropout(dropout)

#     def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, C, H, W) -> (B, N, C)
#         """
#         B, C, H, W = x.shape
#         return x.permute(0, 2, 3, 1).reshape(B, H * W, C)

#     def _to_map(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         """
#         x: (B, N, C) -> (B, C, 1, H, W)
#         """
#         B, N, C = x.shape
#         return x.reshape(B, H, W, C).permute(0, 3, 1, 2).unsqueeze(2)

#     def forward(self, ref_pose: torch.Tensor, driven_pose: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
#         assert ref_pose.dim() == 5 and driven_pose.dim() == 5
#         B, C, one, H, W = ref_pose.shape
#         assert one == 1, "ref_pose must have T=1 at dim=2"
#         Bd, Cd, T, Hd, Wd = driven_pose.shape
#         assert (B, C, H, W) == (Bd, Cd, Hd, Wd), "ref/driven must match B,C,H,W"

#         ref_hw = ref_pose[:, :, 0]  # (B,C,H,W)

#         base = driven_pose[:, :, 0]  # (B,C,H,W) for delta baseline
#         outs: List[torch.Tensor] = []
#         attn_maps: List[torch.Tensor] = []

#         # Precompute Q from ref (same for all t)
#         q = self._to_tokens(ref_hw)        # (B,N,C)
#         q = self.norm_q(q)
#         q = self.q_proj(q)                # (B,N,C)
#         q = q.view(B, -1, self.heads, self.head_dim).transpose(1, 2)  # (B,heads,N,dh)

#         for t in range(T):
#             drv_hw = driven_pose[:, :, t]  # (B,C,H,W)

#             if self.use_delta:
#                 drv_hw = drv_hw - base     # (B,C,H,W)

#             kv = self._to_tokens(drv_hw)   # (B,N,C)
#             kv = self.norm_kv(kv)

#             k = self.k_proj(kv)
#             v = self.v_proj(kv)

#             k = k.view(B, -1, self.heads, self.head_dim).transpose(1, 2)  # (B,heads,N,dh)
#             v = v.view(B, -1, self.heads, self.head_dim).transpose(1, 2)  # (B,heads,N,dh)

#             # attn: (B,heads,N,N)
#             attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)

#             # out: (B,heads,N,dh) -> (B,N,C)
#             out = torch.matmul(attn, v)
#             out = out.transpose(1, 2).contiguous().view(B, -1, C)
#             out = self.out_proj(out)
#             out = self.out_drop(out)

#             out_map = self._to_map(out, H, W)  # (B,C,1,H,W)

#             if self.residual_inject:
#                 # keep ref as anchor; inject motion residual
#                 out_map = ref_pose + out_map   # (B,C,1,H,W)

#             outs.append(out_map)

#             if self.return_attn:
#                 attn_maps.append(attn.detach())

#         aligned_seq = torch.cat(outs, dim=2)  # (B,C,T,H,W)
#         return aligned_seq



import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

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
        # import pudb; pudb.set_trace()
        # 4. Attention 计算: (B*T, heads, N_q, N_kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn_map = attn.detach().cpu().float().numpy()
        # np.save('attn_map.npy', attn_map)
        # import pudb; pudb.set_trace()
        # 5. 输出投影与还原
        out = torch.matmul(attn, v) # (B*T, heads, N, dh)
        out = out.transpose(1, 2).contiguous().reshape(B * T, H * W, C)
        out = self.out_drop(self.out_proj(out))
        # merge = out.detach().cpu().float().numpy()
        # np.save('attn_merge.npy', merge)
        # (B*T, N, C) -> (B, C, T, H, W)
        aligned_seq = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        # drive = driven_pose.detach().cpu().float().numpy()
        # np.save('attn_drive.npy', drive)
        if self.residual_inject:
            # 残差连接在驱动姿态上
            aligned_seq = aligned_seq + driven_pose
        # align = aligned_seq.detach().cpu().float().numpy()
        # np.save('attn_align.npy', align)
        return aligned_seq   ##, attn if self.return_attn else None


class Ref_FrameWiseSpatialCrossAttn(nn.Module):
    """
    互换后的 Cross-Attention 模块:
    Q: Ref Pose (查询者)
    K, V: Driven Pose (被查询的序列特征)
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

        # 注册位置编码
        pos_embed = self._init_2d_pos_embed(dim, height, width)
        self.register_buffer("pos_embed", pos_embed) # (1, H*W, C)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def _init_2d_pos_embed(self, dim, h, w):
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
        ], dim=1) 
        return pos_emb.unsqueeze(0)

    def forward(self, ref_pose: torch.Tensor, driven_pose: torch.Tensor) -> torch.Tensor:
        """
        ref_pose:    (B, C, 1, H, W) -> 现在的 Q
        driven_pose: (B, C, T, H, W) -> 现在的 K, V
        """
        B, C, _, H, W = ref_pose.shape
        _, _, T, _, _ = driven_pose.shape
        target_dtype = ref_pose.dtype
        curr_pos_embed = self.pos_embed.to(target_dtype)

        # --- 1. 处理 Q (来自 Reference) ---
        # (B, C, 1, H, W) -> (B, N, C)
        q_tokens = ref_pose[:, :, 0].permute(0, 2, 3, 1).reshape(B, H * W, C)
        q_tokens = self.norm_q(q_tokens)
        
        # 加上位置编码并扩展到 T 维度，以便与驱动序列的每一帧做 Cross
        q = self.q_proj(q_tokens + curr_pos_embed) 
        q = q.view(B, H * W, self.heads, self.head_dim).transpose(1, 2) # (B, heads, N, dh)
        # 广播 Q 到 (B*T, heads, N, dh)
        q = q.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, self.heads, H * W, self.head_dim)

        # --- 2. 处理 K, V (来自 Driven Sequence) ---
        # (B, C, T, H, W) -> (B*T, N, C)
        kv_tokens = driven_pose.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        kv_tokens = self.norm_kv(kv_tokens)
        
        # K 加入位置编码
        k = self.k_proj(kv_tokens + curr_pos_embed)
        k = k.view(B * T, H * W, self.heads, self.head_dim).transpose(1, 2) # (B*T, heads, N, dh)
        
        v = self.v_proj(kv_tokens).view(B * T, H * W, self.heads, self.head_dim).transpose(1, 2)

        # --- 3. Attention 计算 ---
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_map = attn.detach().cpu().float().numpy()
        np.save('attn_map.npy', attn_map)
        # import pudb; pudb.set_trace()
        # --- 4. 输出投影与还原 ---
        out = torch.matmul(attn, v) # (B*T, heads, N, dh)
        out = out.transpose(1, 2).contiguous().reshape(B * T, H * W, C)
        out = self.out_drop(self.out_proj(out))
        attn_merge = out.detach().cpu().float().numpy()
        np.save('attn_merge.npy', attn_merge)
        # import pudb; pudb.set_trace()
        # 重塑回 (B, C, T, H, W)
        aligned_seq = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        if self.residual_inject:
            # 此时 aligned_seq 的结构源自 Q(Ref)，内容源自 V(Driven)
            # 按照互换逻辑，残差通常加在原始驱动序列上以保持运动流
            aligned_seq = aligned_seq + driven_pose

        return aligned_seq
    
class WanCustom(torch.nn.Module):
    def __init__(self, in_channels=32, out_channels=16, patch_size=[1, 2, 2]):
        super().__init__()
        self.pose_patch_embedding = nn.Conv3d(16, 1536, kernel_size=patch_size, stride=patch_size)

        
        torch.nn.init.xavier_normal_(self.pose_patch_embedding.weight)
        torch.nn.init.zeros_(self.pose_patch_embedding.bias)
        self.align_model = FrameWiseSpatialCrossAttn(dim=1536, heads=12, height=52, width=30, residual_inject=True, return_attn=False)
        # self.align_model = FrameWiseSpatialCrossAttn(dim=1536, heads=12, height=80, width=45, residual_inject=True, return_attn=False)

        
    def patchify(self, x: torch.Tensor):
        x = self.pose_patch_embedding(x)
        # x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x 
    
    def motion_patchify(self, x: torch.Tensor):
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x

    