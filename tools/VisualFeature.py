import argparse
import numpy as np
from lib.models.pose_higher_hrnet import PoseHigherResolutionNet
from torchvision import transforms
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
from lib.config import cfg, update_config, check_config
import yaml
import torch.nn.functional as F

def setup_model(cfg):
    net = PoseHigherResolutionNet(cfg)
    net.load_state_dict(torch.load('D:\MyData\PythonProject\Higherhrnet-paper\output\coco_kpt\pose_higher_hrnet\w32_512_adam_lr1e-3\model_best.pth.tar'))
    net.eval()
    return net


def preprocess_image(path):
    transform_test = transforms.Compose([transforms.Resize((512, 512)),
                                         transforms.ToTensor()])
    img = Image.open(path).convert('RGB')
    img_tensor = transform_test(img)
    img_tensor.unsqueeze_(0)
    return img_tensor


# def show_feature(feature, title):
#     feature_grid = torchvision.utils.make_grid(feature)
#     feature_np = feature_grid.detach().numpy().transpose((1, 2, 0))
#     if len(feature_np.shape) == 3 and feature_np.shape[2] > 3:
#         feature_np = feature_np[:, :, 0]  # 只选择第一个通道
#     feature_np = np.clip(feature_np, 0, 1)
#     plt.imshow(feature_np)
#     plt.title(title)
#     plt.pause(0.001)

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def show_feature(feature, title):
    feature_grid = torchvision.utils.make_grid(feature)
    feature_np = feature_grid.detach().numpy().transpose((1, 2, 0))
    if len(feature_np.shape) == 3 and feature_np.shape[2] > 3:
        # 相加融合
        feature_np = np.sum(feature_np, axis=2)
        # 归一化到0-1范围
        feature_np = (feature_np - feature_np.min()) / (feature_np.max() - feature_np.min())
    feature_np = np.clip(feature_np, 0, 1)
    plt.imshow(feature_np)
    plt.title(title)
    plt.pause(0.001)


def test_one(net, path):
    img_tensor1 = preprocess_image(path)
    final_outputs, high_res_feature_stage2, high_res_feature_stage3, high_res_feature_stage4= net(img_tensor1)
    # final_outputs= net(img_tensor1)
    show_feature(high_res_feature_stage2, 'Stage2 High Res Feature')
    show_feature(high_res_feature_stage3, 'Stage3 High Res Feature')
    show_feature(high_res_feature_stage4, 'Stage4 High Res Feature')
    show_feature(final_outputs[-1], 'Final High Res Feature')
    show_heatmap(high_res_feature_stage2, 'Stage2 High Res Feature Heatmap')
    show_heatmap(high_res_feature_stage3, 'Stage3 High Res Feature Heatmap')
    show_heatmap(high_res_feature_stage3, 'Stage4 High Res Feature Heatmap')
    show_heatmap( final_outputs[0][:, :4, :, :], 'HRNet Heatmap')
    show_heatmap(final_outputs[-1], 'HigherHRNet Heatmap')
    print(final_outputs[0][:, :4, :, :].shape,final_outputs[-1].shape)
    # 为了使尺寸相同，需要对final_outputs[0]进行插值
    # 加入插值代码
    interpolated_output = F.interpolate(
        final_outputs[0][:, :4, :, :],
        size=(final_outputs[-1].size(2), final_outputs[-1].size(3)),
        mode='bilinear',
        align_corners=False
    )
    print("转换后",interpolated_output.shape)
    average_heatmap = ( interpolated_output + final_outputs[-1]) / 2
    show_heatmap(average_heatmap, ' Final Heatmap')
    print(average_heatmap)
    # 调用新函数将average_heatmap生成到原图片上
    path="E:\\coco\\images\\val2017\\435.jpg"
    # 将average_heatmap展示在原图片上
    show_all_channels_heatmap_on_image(path, average_heatmap)
    # 获取 final_outputs[0] 中后 4 个通道的 tag
    tag = final_outputs[0][:, -4:, :, :]
    print(tag)
    output1=final_outputs[-1]
    print("Final Output Shape:",final_outputs[0].shape,output1.shape)
# def show_feature(feature, title):
#     feature1=feature.transpose(1,0)
#     feature_grid = torchvision.utils.make_grid(feature1)
#     # feature_np = feature_grid.detach().numpy().transpose((1, 2, 0))
#     # if len(feature_np.shape) == 3 and feature_np.shape[2] > 3:
#     #     # 相加融合
#     #     feature_np = np.sum(feature_np, axis=2)
#     #     # 归一化到0-1范围
#     #     feature_np = (feature_np - feature_np.min()) / (feature_np.max() - feature_np.min())
#     # feature_np = np.clip(feature_np, 0, 1)
#     inp = feature_grid.detach().numpy().transpose((1, 2, 0))  # 将通道数放在最后一维
#     feature_np = np.clip(inp, 0, 1)
#     plt.imshow(feature_np)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# def show_heatmap(feature, title):
#     # 对特征图进行一些处理以得到适合显示的热图
#     # 这里简单地对特征图取平均值得到单通道
#     if len(feature.shape) == 4:
#         feature = np.mean(feature.detach().cpu().numpy(), axis=1)
#     elif len(feature.shape) == 3:
#         feature = np.mean(feature.detach().cpu().numpy(), axis=0)
#     # 去掉多余的维度
#     if len(feature.shape) == 3 and feature.shape[0] == 1:
#         feature = feature.squeeze(0)
#     # 归一化到0-1范围
#     feature = (feature - feature.min()) / (feature.max() - feature.min())
#     plt.imshow(feature, cmap='hot')
#     plt.title(title)
#     plt.pause(0.001)
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def show_heatmap(feature, title_prefix):
    if len(feature.shape) == 4:
        # 处理4D张量（通常是[batch_size, channels, height, width]）
        feature = feature.detach().cpu().numpy()
        num_channels = feature.shape[1]
        num_cols = min(4, num_channels)  # 每行最多显示4个热图
        num_rows = (num_channels + num_cols - 1) // num_cols
        plt.figure(figsize=(num_cols * 3, num_rows * 3))
        for i in range(num_channels):
            channel_feature = feature[0, i, :, :]  # 假设取batch中的第一个样本
            # 归一化到0-1范围
            channel_feature = (channel_feature - channel_feature.min()) / (channel_feature.max() - channel_feature.min())
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(channel_feature, cmap='hot')
            plt.title(f"{title_prefix}_channel_{i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    elif len(feature.shape) == 3:
        # 处理3D张量（通常是[channels, height, width]）
        feature = feature.detach().cpu().numpy()
        num_channels = feature.shape[0]
        num_cols = min(4, num_channels)  # 每行最多显示4个热图
        num_rows = (num_channels + num_cols - 1) // num_cols
        plt.figure(figsize=(num_cols * 3, num_rows * 3))
        for i in range(num_channels):
            channel_feature = feature[i, :, :]
            # 归一化到0-1范围
            channel_feature = (channel_feature - channel_feature.min()) / (channel_feature.max() - channel_feature.min())
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(channel_feature, cmap='hot')
            plt.title(f"{title_prefix}_channel_{i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def show_all_channels_heatmap_on_image(image_path, heatmap):
    # 加载原图片
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    heatmap = heatmap.detach().cpu().numpy()[0]  # 假设取batch中的第一个样本
    num_channels = heatmap.shape[0]

    plt.figure(figsize=(15, 5))
    for channel_idx in range(num_channels):
        channel_heatmap = heatmap[channel_idx]
        channel_heatmap = channel_heatmap.squeeze()
        channel_heatmap = (channel_heatmap - channel_heatmap.min()) / (channel_heatmap.max() - channel_heatmap.min())
        channel_heatmap = np.uint8(channel_heatmap * 255)
        channel_heatmap = Image.fromarray(channel_heatmap).convert('L')
        channel_heatmap = channel_heatmap.resize(img.shape[:2][::-1], Image.ANTIALIAS)
        channel_heatmap = np.array(channel_heatmap)
        channel_heatmap = np.stack([channel_heatmap] * 3, axis=2)

        overlay = np.uint8(0.5 * img + 0.5 * channel_heatmap)

        plt.subplot(1, num_channels, channel_idx + 1)
        plt.imshow(overlay)
        plt.title(f'Channel {channel_idx + 1} Heatmap on Original Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = argparse.Namespace(cfg='D:\MyData\PythonProject\Higherhrnet-paper\experiments\coco\higher_hrnet\w32_512_adam_lr1e-3.yaml', opts=[])
    update_config(cfg, args)
    check_config(cfg)
    net = setup_model(cfg)
    test_image_path = "E:\\coco\\images\\val2017\\435.jpg"
    test_one(net, test_image_path)
