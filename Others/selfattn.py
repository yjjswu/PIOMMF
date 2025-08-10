import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_dim=None):
        """
        自注意力机制
        Args:
            input_dim (int): 输入特征维度 C
            head_dim (int, optional): 注意力头维度，默认使用 input_dim // 8
        """
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim or input_dim // 8
        
        # 定义Q/K/V变换矩阵
        self.query = nn.Linear(input_dim, self.head_dim)
        self.key = nn.Linear(input_dim, self.head_dim)
        self.value = nn.Linear(input_dim, input_dim)  # 保持输出维度与输入一致
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
        # 输出投影（保持维度一致）
        self.out_proj = nn.Linear(input_dim, input_dim)
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for module in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 形状为 (N, C) 的输入张量
        Returns:
            Tensor: 形状为 (N, C) 的输出张量
        """
        N, C = x.shape
        
        # 计算Q/K/V [N, head_dim]
        q = self.query(x)   # (N, head_dim)
        k = self.key(x)     # (N, head_dim)
        v = self.value(x)   # (N, C)
        
        # 计算注意力分数 [N, N]
        attn_scores = torch.matmul(q, k.transpose(0, 1)) * self.scale  # (N, N)
        
        # 应用softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, N)
        
        # 加权求和 [N, C]
        output = torch.matmul(attn_weights, v)  # (N, C)
        
        # 输出投影
        output = self.out_proj(output)  # (N, C)
        
        return output

# 使用示例
if __name__ == "__main__":
    # 创建输入张量 (5个元素，每个64维)
    x = torch.randn(20, 64)
    
    # 初始化自注意力模块
    sa = SelfAttention(input_dim=64)
    
    # 前向计算
    output = sa(x)
    
    print("输入形状:", x.shape)      # torch.Size([5, 64])
    print("输出形状:", output.shape) # torch.Size([5, 64])