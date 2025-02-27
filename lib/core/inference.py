# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
#在人体关键点检测任务中，从模型获取输出、处理多阶段输出以及聚合结果的功能
# 从dataset.transforms模块导入FLIP_CONFIG，用于处理图像翻转相关配置
from dataset.transforms import FLIP_CONFIG

# 在人体关键点检测任务中，从模型获取输出、处理多阶段输出以及聚合结果的功能

# 获取模型输出、热图和标签
def get_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    # 初始化输出列表、热图列表和标签列表
    outputs = []
    heatmaps = []
    tags = []

    # 将模型对输入图像的输出添加到outputs列表
    outputs.append(model(image))
    # 从模型输出中提取热图部分，热图包含关键点的概率分布信息
    '''outputs[-1]：选取 outputs 列表中的最后一个元素，这个元素就是模型的最新输出张量。[:, :cfg.DATASET.NUM_JOINTS]：这是一个切片操作。: 表示选取所有的 batch_size 维度，即对批量中的所有样本进行操作。
:cfg.DATASET.NUM_JOINTS 表示选取从第 0 个通道到第 cfg.DATASET.NUM_JOINTS - 1 个通道。这里 cfg.DATASET.NUM_JOINTS 是配置文件中定义的关键点数量，这意味着提取的通道数与关键点数量相同，这些通道对应着每个关键点的热图信息，每个通道的值表示对应关键点在图像上每个位置的概率分布。
heatmaps.append(...)：将提取出来的热图张量添加到 heatmaps 列表中。'''
    heatmaps.append(outputs[-1][:, :cfg.DATASET.NUM_JOINTS])
    # 从模型输出中提取标签部分，标签用于后续的分组等操作
    '''[:, cfg.DATASET.NUM_JOINTS:]：这也是一个切片操作。
: 选取所有的 batch_size 维度。
cfg.DATASET.NUM_JOINTS: 表示从第 cfg.DATASET.NUM_JOINTS 个通道开始，选取到最后一个通道。这部分通道包含的信息就是标签，在后续处理中可能用于关键点的分组等操作。'''
    tags.append(outputs[-1][:, cfg.DATASET.NUM_JOINTS:])

    # 如果需要进行图像翻转增强
    if with_flip:
        # 将翻转后的图像输入模型，得到翻转后的输出
        outputs.append(model(torch.flip(image, [3])))
        # 对翻转后的输出进行翻转恢复，以保证与原始图像方向一致
        outputs[-1] = torch.flip(outputs[-1], [3])
        # 从翻转后的输出中提取热图
        heatmaps.append(outputs[-1][:, :cfg.DATASET.NUM_JOINTS])
        # 从翻转后的输出中提取标签
        tags.append(outputs[-1][:, cfg.DATASET.NUM_JOINTS:])

        # 根据数据集名称确定数据集类型
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            # 如果是新的数据集，需要实现对应的翻转索引
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)

        # 获取对应的翻转索引
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]
        # 对翻转后的热图进行索引调整，以匹配原始图像的关键点顺序
        heatmaps[-1] = heatmaps[-1][:, flip_index, :, :]
        # 如果每个关节点都有独立标签，对翻转后的标签进行索引调整
        if cfg.MODEL.TAG_PER_JOINT:
            tags[-1] = tags[-1][:, flip_index, :, :]

    # 如果数据集中包含中心关键点且在测试时忽略中心关键点
    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        # 去掉热图中的中心关键点部分
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        # 去掉标签中的中心关键点部分
        tags = [tms[:, :-1] for tms in tags]

    # 如果需要将热图和标签投影到特定尺寸的图像上
    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    # 返回模型输出、热图和标签
    return outputs, heatmaps, tags


# 获取多阶段模型输出、热图和标签
def get_multi_stage_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    # 初始化热图平均值和热图数量
    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []
    tags = []

    # 获取模型的多阶段输出
    outputs = model(image)
    for i, output in enumerate(outputs):
        # 如果有多阶段输出且不是最后一阶段
        if len(outputs) > 1 and i!= len(outputs) - 1:
            # 对中间阶段的输出进行插值，使其尺寸与最后一阶段相同
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=False
            )

        # 计算特征偏移量，用于确定标签的起始位置
        offset_feat = cfg.DATASET.NUM_JOINTS \
            if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0

        # 如果当前阶段需要计算热图损失且在测试中使用热图
        if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
            # 累加热图，累加热图的形状依然是 (N, C, H, W)。因为在累加操作中，是对应位置的元素相加，不会改变热图的维度结构。[1,4,256,256]
            heatmaps_avg += output[:, :cfg.DATASET.NUM_JOINTS]
            # 热图数量加一
            num_heatmaps += 1

        # 如果当前阶段需要计算AE损失且在测试中使用AE，只是在Hrnet结果中取
        if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
            # 提取标签
            tags.append(output[:, offset_feat:])

    # 如果有累加的热图,上面的最终结果是将
    if num_heatmaps > 0:
        # 计算平均热图,每个位置的平均值，从而得到平均热图
        heatmaps.append(heatmaps_avg / num_heatmaps)

    # 如果需要进行图像翻转增强
    if with_flip:
        # 根据数据集名称确定数据集类型
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            # 如果是新的数据集，需要实现对应的翻转索引
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)

        # 获取对应的翻转索引
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]

        # 初始化热图平均值和热图数量
        heatmaps_avg = 0
        num_heatmaps = 0
        # 将翻转后的图像输入模型，得到多阶段的翻转输出
        outputs_flip = model(torch.flip(image, [3]))
        for i in range(len(outputs_flip)):
            output = outputs_flip[i]
            # 如果有多阶段输出且不是最后一阶段
            if len(outputs_flip) > 1 and i!= len(outputs_flip) - 1:
                # 对中间阶段的输出进行插值，使其尺寸与最后一阶段相同
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            # 对翻转后的输出进行翻转恢复，以保证与原始图像方向一致
            output = torch.flip(output, [3])
            # 将翻转后的输出添加到总输出列表
            outputs.append(output)

            # 计算特征偏移量，用于确定标签的起始位置
            offset_feat = cfg.DATASET.NUM_JOINTS \
                if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0

            # 如果当前阶段需要计算热图损失且在测试中使用热图
            if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
                # 累加翻转后的热图并进行索引调整
                heatmaps_avg += \
                    output[:, :cfg.DATASET.NUM_JOINTS][:, flip_index, :, :]
                # 热图数量加一
                num_heatmaps += 1

            # 如果当前阶段需要计算AE损失且在测试中使用AE
            if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
                # 提取翻转后的标签
                tags.append(output[:, offset_feat:])
                # 如果每个关节点都有独立标签，对翻转后的标签进行索引调整
                if cfg.MODEL.TAG_PER_JOINT:
                    tags[-1] = tags[-1][:, flip_index, :, :]

        # 计算翻转后的平均热图
        heatmaps.append(heatmaps_avg / num_heatmaps)

    # 如果数据集中包含中心关键点且在测试时忽略中心关键点，
    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        # 去掉热图中的中心关键点部分
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        # 去掉标签中的中心关键点部分
        tags = [tms[:, :-1] for tms in tags]

    # 如果需要将热图和标签投影到特定尺寸的图像上
    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    # 返回多阶段模型输出、热图和标签
    return outputs, heatmaps, tags


# 聚合模型输出结果，
def aggregate_results(
        cfg, scale_factor, final_heatmaps, tags_list, heatmaps, tags
):
    # 如果缩放因子为1或只有一个缩放因子
    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        # 如果最终热图已存在且不需要投影到图像，对 tags 中的每个标签张量 tms 使用双线性插值方法调整其尺寸，使其与 final_heatmaps 的高度和宽度一致。
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        # 将每个标签添加到标签列表中，给标签加一个维度
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    # 计算平均热图（如果进行了翻转测试，上面含函数处理的热图列表包含两个一个是原始的一个是翻转的）
    heatmaps_avg = (heatmaps[0] + heatmaps[1]) / 2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]

    # 如果最终热图为空
    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    # 如果需要投影到图像
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    # 否则对平均热图进行插值并累加
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )
    # 返回最终热图和标签列表tags_list 是一个包含处理后标签的列表，其中每个标签张量的形状为 (N, C, H, W, 1)
    return final_heatmaps, tags_list