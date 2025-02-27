from lib.models.pose_higher_hrnet import PoseHigherResolutionNet
from lib.config import cfg, update_config, check_config
import torch
import argparse
# 初始化模型
args = argparse.Namespace(cfg='D:\MyData\PythonProject\Higherhrnet-paper\experiments\coco\higher_hrnet\w32_512_adam_lr1e-3.yaml', opts=[])
update_config(cfg, args)
check_config(cfg)
model = PoseHigherResolutionNet(cfg)
model.init_weights()

# 随机输入
x = torch.randn(1, 3, 512, 512)
outputs = model(x)

# 提取标签
heatmaps = outputs[0][:, :cfg.MODEL.NUM_JOINTS]
tags = outputs[0][:, cfg.MODEL.NUM_JOINTS:]

print("标签值的范围：", tags.min().item(), "~", tags.max().item())
print("标签均值：", tags.mean().item())