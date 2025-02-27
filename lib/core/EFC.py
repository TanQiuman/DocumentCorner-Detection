import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：A Lightweight Fusion Strategy With Enhanced Interlayer Feature Correlation for Small Object Detection
# 中文题目:  轻量级融合策略增强层间特征相关性，用于小目标检测
# 论文链接：https://ieeexplore.ieee.org/abstract/document/10671587
# 官方github：https://github.com/nuliweixiao/EFC

class EFC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        # 第一层卷积，将输入通道数 c1 转换为 c2
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)

        # 批归一化层
        self.bn = nn.BatchNorm2d(c2)

        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

        # 组数和常数 epsilon
        self.group_num = 16
        self.eps = 1e-10

        # 可学习的参数 gamma 和 beta，用于后续的归一化
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))

        # 门控生成器：用于生成权重
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )

        # 深度可分卷积（Depthwise Convolution）
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)

        # 其他卷积层
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)

        # 其他的常数
        self.one = c2
        self.two = c2

        # 用于全局信息的卷积层
        self.conv4_global = nn.Conv2d(c2, 1, kernel_size=1, stride=1)

        # 交互卷积层
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1)

    def forward(self, x):
        # 输入的 x 包含两个部分 x1 和 x2
        x1, x2 = x

        # 第一个卷积：对 x1 进行卷积并归一化
        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigmoid(bn_x)

        # 第二个卷积：对 x2 进行卷积并归一化
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigmoid(bn_x2)

        # 全局特征融合
        X_GLOBAL = global_conv1 + global_conv2

        # 使用 1x1 卷积处理全局信息
        x_conv4 = self.conv4_global(X_GLOBAL)
        X_4_sigmoid = self.sigmoid(x_conv4)

        # 对全局信息进行加权
        X_ = X_4_sigmoid * X_GLOBAL

        # 分块操作
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()

            # 计算均值并归一化
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1

            # Softmax 操作
            x_2_2 = F.softmax(x_1_av, dim=1)
            x1 = x_2_2.reshape(N, C, H, W)

            # 加权后进行合并
            x1 = X_[group_id] * x1
            out.append(x1)

        # 合并所有分支的结果
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)

        N, C, H, W = out.size()

        # 重新排列和计算均值
        x_add_1 = out.reshape(N, self.group_num, -1)
        x_shape_1 = X_GLOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)

        # 标准化操作
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)

        # 最终的加权输出
        x_gui = (x_guiyi_1 * self.gamma + self.beta)

        # 权重生成与条件判断
        weight_x3 = self.Apt(X_GLOBAL)
        reweights = self.sigmoid(weight_x3)

        # 根据生成的权重决定加权过程
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2

        x_up = x_up_1 * X_GLOBAL + x_up_2 * X_GLOBAL
        x_low = x_low_1 * X_GLOBAL + x_low_2 * X_GLOBAL

        # 深度卷积和交互操作
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)

        # 门控生成操作
        x_so = self.gate_generator(x_low)
        x11_up_dwc = x11_up_dwc * x_so

        # 最终输出
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw
        xL = xL + x_gui

        return xL


if __name__ == '__main__':
    # 输入数据（批大小为 1，通道数为 32，图像大小为 256x256）
    x1 = torch.randn(1, 32, 256, 256)
    x2 = torch.randn(1, 32, 256, 256)
    x = (x1, x2)

    # 创建模型实例
    model = EFC(32, 32)

    # 打印输出的形状
    print(model(x).shape)
