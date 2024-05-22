import torch
import torch.nn as nn

class SAHIModule(nn.Module):
    def __init__(self, input_channels, output_channels, num_slices, slice_factor=0.5):
        super(SAHIModule, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.slice_factor = slice_factor

        # 计算每个切片的通道数
        self.slice_channels = int(input_channels * slice_factor)

        # 创建用于每个切片的卷积层
        self.slice_conv = nn.ModuleList([nn.Conv2d(self.slice_channels, self.slice_channels, kernel_size=3, padding=1) for _ in range(self.num_slices)])

        # 用于合并切片的卷积层
        self.merge_conv = nn.Conv2d(self.slice_channels * self.num_slices, output_channels, kernel_size=1)

    def forward(self, x):
        # 将输入切分为切片
        slices = torch.split(x, self.slice_channels, dim=1)
        processed_slices = []

        # 用单独的卷积层处理每个切片
        for i, slice in enumerate(slices):
            processed_slice = self.slice_conv[i](slice)
            processed_slices.append(processed_slice)

        # 合并处理后的切片
        merged_slices = torch.cat(processed_slices, dim=1)

        # 应用最终的卷积以合并切片
        output = self.merge_conv(merged_slices)

        return output

# 示例：创建一个适应YOLOv5s的SAHI模块
sa_module = SAHIModule(input_channels=256, output_channels=256, num_slices=4, slice_factor=0.5)
