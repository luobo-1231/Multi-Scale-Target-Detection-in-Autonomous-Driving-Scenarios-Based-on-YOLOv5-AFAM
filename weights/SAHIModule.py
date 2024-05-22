import torch
import torch.nn as nn

class SAHIModule(nn.Module):
    def __init__(self, channels, intermediate_channels, num_slices):
        super(SAHIModule, self).__init__()

        self.channels = channels
        self.num_slices = num_slices
        self.slice_channels = intermediate_channels

        # 创建用于每个切片的卷积层
        self.slice_conv = nn.ModuleList(
            [nn.Conv2d(channels, self.slice_channels, kernel_size=3, padding=1) for _ in range(self.num_slices)])

        # 用于合并切片的卷积层
        # 输入通道数现在是 slice_channels * num_slices
        self.merge_conv = nn.Conv2d(self.slice_channels * self.num_slices, channels, kernel_size=1)

    def forward(self, x):
        processed_slices = []  # 用于存储每个切片的处理结果
        print(f"Number of slices: {len(self.slice_conv)}")
        for i, conv in enumerate(self.slice_conv):
            print(f"Processing slice {i}")
            processed_slice = conv(x)
            processed_slices.append(processed_slice)  # 将每个切片的处理结果添加到列表中
        merged_slices = torch.cat(processed_slices, dim=1)  # 合并所有切片的处理结果
        output = self.merge_conv(merged_slices)
        return output

# 示例
sa_module = SAHIModule(channels=64, intermediate_channels=32, num_slices=2)

# 创建一个示例输入张量，可以替换为你的实际输入数据
input_tensor = torch.randn(1, 64, 256, 256)

# 调用模块的forward方法
output_tensor = sa_module(input_tensor)

# 打印输出张量的形状
print("Output shape:", output_tensor.shape)
