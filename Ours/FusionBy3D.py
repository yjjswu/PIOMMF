import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewFusion(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, H=448, W=448):
        super(MultiViewFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = H
        self.W = W
        # 定义特征提取模块
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 定义3D特征体投影和聚合模块
        self.proj_3d = nn.Linear(out_channels, out_channels)
        self.aggregate_3d = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(2,1,1),padding=(0,1,1))
        # 定义空间增强注意力机制
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels , out_channels , kernel_size=1),
            nn.Softmax(dim=1)
        )
    def forward(self, front_view, back_view, side_view, K_front, K_back, K_side, RT_front, RT_back, RT_side):
        # 前向传播
        batch_size = front_view.size(0)
        # 特征提取
        front_feat = self.feature_extractor(front_view)
        back_feat = self.feature_extractor(back_view)
        side_feat = self.feature_extractor(side_view)
        # 按角度投射到3D特征体
        front_3d = self.project_to_3d(front_feat, K_front, RT_front)
        back_3d = self.project_to_3d(back_feat, K_back, RT_back)
        side_3d = self.project_to_3d(side_feat, K_side, RT_side)
        # 聚合3D特征
        fused_3d = torch.stack([front_3d, back_3d, side_3d], dim=2)  # shape: [N, C, 3, H, W]
        # print(fused_3d.shape)
        # fused_3d = self.aggregate_3d(fused_3d).view(batch_size, self.out_channels * 3, self.H,
        #                                             self.W)  # shape: [N, C, H, W]
        fused_3d=self.aggregate_3d(fused_3d).view(batch_size, self.out_channels, self.H,self.W)
        # fused_3d = torch.cat(fused_3d,dim)
        # 空间增强注意力
        # 调整形状以匹配卷积层的输入要求
        # print(fused_3d.shape)
        attn = self.spatial_attn(fused_3d)
        fused_feat = fused_3d * attn
        return fused_feat
    def project_to_3d(self, feat_2d, K, RT):
        # 根据相机参数将2D特征投射到3D空间
        B, C, H, W = feat_2d.shape
        grid = self.create_2d_grid(H, W, B)
        feat_3d = F.grid_sample(feat_2d, grid, align_corners=True)
        return feat_3d
    def create_2d_grid(self, H, W, batch_size):
        # 创建2D网格
        x = torch.linspace(-1, 1, W).cuda()
        y = torch.linspace(-1, 1, H).cuda()
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.expand(batch_size, H, W, 2)
        return grid


# 测试代码
if __name__ == "__main__":
    # 创建随机输入数据
    batch_size = 4
    H, W = 448, 448
    front_view = torch.randn(batch_size, 1, H, W)
    back_view = torch.randn(batch_size, 1, H, W)
    side_view = torch.randn(batch_size, 1, H, W)

    # 相机内参（假设所有相机的内参相同）
    fx = 224.0  # 焦距（x方向）
    fy = 224.0  # 焦距（y方向）
    cx = 224.0  # 图像中心（x方向）
    cy = 224.0  # 图像中心（y方向）
    # 创建相机参数
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32).repeat(batch_size, 1, 1)
    RT_front = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float32).repeat(batch_size, 1, 1)
    RT_back = torch.tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0]
    ], dtype=torch.float32).repeat(batch_size, 1, 1)
    RT_side = torch.tensor([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0]
    ], dtype=torch.float32).repeat(batch_size, 1, 1)
    # 创建模型实例
    model = MultiViewFusion(in_channels=1, out_channels=64, H=H, W=W)
    # 前向传播
    output = model(front_view, back_view, side_view, K, K, K, RT_front, RT_back, RT_side)
    # 打印输出形状
    print("Output shape:", output.shape)

