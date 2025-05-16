import torch
import torch.nn as nn
import math
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_features, depth_features):
        # 让 RGB 作为 Query，Depth 作为 Key-Value
        attn_output, _ = self.multihead_attn(query=rgb_features, key=depth_features, value=depth_features)
        return self.norm(attn_output + rgb_features)  # 残差连接

class CrossAttentionLora(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 分别定义四个线性层，用以支持 LoRA 注入
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_features, depth_features):
        # 让 rgb 做 Query，depth 做 Key/Value
        B, N, D = rgb_features.size()  
        _, M, _ = depth_features.size()

        # Q, K, V 计算
        Q = self.q_proj(rgb_features)  # [B, N, D]
        K = self.k_proj(depth_features)  # [B, M, D]
        V = self.v_proj(depth_features)  # [B, M, D]

        # 分多头
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        K = K.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]
        V = V.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B, heads, N, M]
        attn_weights = attn_scores.softmax(dim=-1)  # [B, heads, N, M]
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ V  # [B, heads, N, head_dim]

        # 拼回原形
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        # 输出投影 + 残差 + LN
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm(attn_output + rgb_features)

        return attn_output