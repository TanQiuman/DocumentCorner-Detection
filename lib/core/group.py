# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# # Munkres算法是匈牙利算法的实现，用于解决最优化分配问题。
# from munkres import Munkres
# import numpy as np
# import torch
#
#
# # py_max_match函数使用匈牙利算法进行最优化匹配
# # 传入的scores是一个二维数组，表示候选关键点之间的相似度分数
# def py_max_match(scores):
#     # 根据标签值的相似性，将候选关键点分配到不同的实例
#     m = Munkres()
#     tmp = m.compute(scores)  # 匈牙利算法求解最优化匹配
#     tmp = np.array(tmp).astype(np.int32)
#     return tmp
#
#
# # match_by_tag函数根据标签值和其他信息进行匹配
# def match_by_tag(inp, params):
#     # 检查params是否为Params类的实例
#     assert isinstance(params, Params), 'params should be class Params()'
#
#     # 输入的参数是一个包含标签、位置和置信度值的元组
#     tag_k, loc_k, val_k = inp
#
#     # 默认初始化，存储每个关节的3个坐标和标签值
#     default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]))
#
#     # 用于存储关节信息的字典
#     joint_dict = {}
#     # 用于存储标签信息的字典
#     tag_dict = {}
#
#     # 遍历每个关节
#     for i in range(params.num_joints):
#         idx = params.joint_order[i]  # 获取关节的索引
#
#         tags = tag_k[idx]  # 标签
#         joints = np.concatenate(
#             (loc_k[idx], val_k[idx, :, None], tags), 1  # 合并位置、置信度值和标签
#         )
#         mask = joints[:, 2] > params.detection_threshold  # 基于置信度值筛选有效的关节
#         tags = tags[mask]
#         joints = joints[mask]
#
#         if joints.shape[0] == 0:
#             continue  # 如果没有有效关节，跳过当前关节
#
#         # 如果是第一个关节或joint_dict为空（表示没有匹配的关节）
#         if i == 0 or len(joint_dict) == 0:
#             for tag, joint in zip(tags, joints):
#                 key = tag[0]  # 使用标签的第一个值作为唯一标识
#                 joint_dict.setdefault(key, np.copy(default_))[idx] = joint
#                 tag_dict[key] = [tag]
#         else:
#             # 如果已经有匹配的关节，则按标签聚类
#             grouped_keys = list(joint_dict.keys())[:params.max_num_people]
#             grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]
#
#             # 如果达到最大人数，跳过当前迭代
#             if params.ignore_too_much and len(grouped_keys) == params.max_num_people:
#                 continue
#
#             # 计算当前关节与已分配关节之间的差异（欧几里得距离）
#             diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
#             diff_normed = np.linalg.norm(diff, ord=2, axis=2)
#             diff_saved = np.copy(diff_normed)
#
#             # 如果使用了置信度值，则对距离进行调整
#             if params.use_detection_val:
#                 diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]
#
#             num_added = diff.shape[0]  # 当前新增关节的数量
#             num_grouped = diff.shape[1]  # 已分配关节的数量
#
#             # 如果新增关节比已分配关节多，则扩展距离矩阵
#             if num_added > num_grouped:
#                 diff_normed = np.concatenate(
#                     (
#                         diff_normed,
#                         np.zeros((num_added, num_added - num_grouped)) + 1e10  # 用一个极大的值填充空缺
#                     ),
#                     axis=1
#                 )
#
#             # 使用匈牙利算法求解最优匹配
#             pairs = py_max_match(diff_normed)
#             for row, col in pairs:
#                 # 如果匹配成功且满足阈值条件
#                 if (
#                         row < num_added
#                         and col < num_grouped
#                         and diff_saved[row][col] < params.tag_threshold
#                 ):
#                     key = grouped_keys[col]
#                     joint_dict[key][idx] = joints[row]  # 更新匹配关节
#                     tag_dict[key].append(tags[row])  # 更新标签
#                 else:
#                     key = tags[row][0]
#                     joint_dict.setdefault(key, np.copy(default_))[idx] = joints[row]
#                     tag_dict[key] = [tags[row]]
#
#     # 将所有匹配的关节返回，结果是一个numpy数组
#     ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
#     return ans
#
#
# # Params类用于设置模型的各种超参数
# class Params(object):
#     def __init__(self, cfg):
#         # self.num_joints = cfg.DATASET.NUM_JOINTS
#         # self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE
#         self.num_joints = 4  # 设置关键点数量
#         self.max_num_people = 5  # 最大人数
#         # self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD
#         # self.tag_threshold = cfg.TEST.TAG_THRESHOLD
#         self.detection_threshold = 0.2  # 检测阈值
#         self.tag_threshold = 1  # 标签匹配阈值
#         self.use_detection_val = cfg.TEST.USE_DETECTION_VAL  # 是否使用置信度值
#         self.ignore_too_much = cfg.TEST.IGNORE_TOO_MUCH  # 是否忽略过多标签
#
#         # 如果配置中包含中心点，并且不忽略中心点，减少关节数量
#         if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
#             self.num_joints -= 1
#
#         # 配置关节的顺序，如果包含中心点，则调整顺序
#         if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
#             self.joint_order = [
#                 i - 1 for i in [5, 1, 2, 3, 4]  # 如果有中心点，将其放在顺序最前面
#             ]
#         else:
#             self.joint_order = [
#                 i - 1 for i in [1, 2, 3, 4]  # 没有中心点时，顺序从1到4
#             ]
# class HeatmapParser(object):
#     def __init__(self, cfg):
#         """
#         初始化热图解析器对象。
#
#         参数:
#         cfg: 配置对象，包含了模型和测试相关的配置参数。
#         """
#         # 初始化参数对象，该对象存储了关节数量、最大人数等参数
#         self.params = Params(cfg)
#         # 标记每个关节是否有独立的标签
#         self.tag_per_joint = cfg.MODEL.TAG_PER_JOINT
#         # 创建一个最大池化层，用于非极大值抑制（NMS）操作
#         # cfg.TEST.NMS_KERNEL 是池化核的大小，1 是步长，cfg.TEST.NMS_PADDING 是填充大小
#         self.pool = torch.nn.MaxPool2d(
#             cfg.TEST.NMS_KERNEL, 1, cfg.TEST.NMS_PADDING
#         )
#
#     def nms(self, det):
#         """
#         对热图进行非极大值抑制（NMS）操作，抑制重叠的候选点，只保留局部极大值点。
#
#         参数:
#         det (torch.Tensor): 输入的热图张量。
#
#         返回:
#         torch.Tensor: 经过非极大值抑制后的热图张量。
#         """
#         # 对输入的热图进行最大池化操作
#         maxm = self.pool(det)
#         # 创建一个掩码，只保留热图中等于最大池化结果的点，即局部极大值点
#         maxm = torch.eq(maxm, det).float()
#         # 将热图与掩码相乘，过滤掉非极大值点
#         det = det * maxm
#         return det
#
#     def match(self, tag_k, loc_k, val_k):
#         """
#         根据标签将检测到的关键点分组到不同的人体实例中。
#
#         参数:
#         tag_k (numpy.ndarray): 关键点的标签数组。
#         loc_k (numpy.ndarray): 关键点的位置数组。
#         val_k (numpy.ndarray): 关键点的置信度数组。
#
#         返回:
#         list: 分组后的关键点列表，每个元素对应一个样本的分组结果。
#         """
#         # 定义一个匿名函数，用于调用 match_by_tag 函数进行匹配
#         match = lambda x: match_by_tag(x, self.params)
#         # 对每个样本的标签、位置和置信度进行匹配，并将结果存储在列表中
#         return list(map(match, zip(tag_k, loc_k, val_k)))
#
#     def top_k(self, det, tag):
#         """
#         从热图中提取每个关节置信度最高的前 max_num_people 个关键点。
#
#         参数:
#         det (torch.Tensor): 输入的热图张量。
#         tag (torch.Tensor): 输入的标签张量。
#
#         返回:
#         dict: 包含关键点的标签、位置和置信度的字典。
#         """
#         # 对热图进行非极大值抑制操作
#         det = self.nms(det)
#         # 获取热图中的样本数量
#         num_images = det.size(0)
#         # 获取热图中的关节数量
#         num_joints = det.size(1)
#         # 获取热图的高度
#         h = det.size(2)
#         # 获取热图的宽度
#         w = det.size(3)
#         # 将热图张量的形状调整为 (num_images, num_joints, -1)
#         det = det.view(num_images, num_joints, -1)
#         # 找出每个关节置信度最高的前 max_num_people 个点，返回置信度和索引
#         val_k, ind = det.topk(self.params.max_num_people, dim=2)
#
#         # 将标签张量的形状调整为 (tag.size(0), tag.size(1), w*h, -1)
#         tag = tag.view(tag.size(0), tag.size(1), w * h, -1)
#         # 如果每个关节没有独立的标签，则将标签张量扩展为每个关节都有相同的标签
#         if not self.tag_per_joint:
#             tag = tag.expand(-1, self.params.num_joints, -1, -1)
#
#         # 从标签张量中提取前 max_num_people 个点的标签
#         tag_k = torch.stack(
#             [
#                 torch.gather(tag[:, :, :, i], 2, ind)
#                 for i in range(tag.size(3))
#             ],
#             dim=3
#         )
#
#         # 计算前 max_num_people 个点的 x 坐标
#         x = ind % w
#         # 计算前 max_num_people 个点的 y 坐标
#         y = (ind / w).long()
#
#         # 将 x 和 y 坐标拼接在一起
#         ind_k = torch.stack((x, y), dim=3)
#
#         # 返回一个字典，包含关键点的标签、位置和置信度
#         ans = {
#             'tag_k': tag_k.cpu().numpy(),
#             'loc_k': ind_k.cpu().numpy(),
#             'val_k': val_k.cpu().numpy()
#         }
#
#         return ans
#
#     def adjust(self, ans, det):
#         """
#         对关键点的位置进行微调，以提高关键点检测的准确性。
#
#         参数:
#         ans (list): 关键点数组列表，每个元素对应一个样本的关键点信息。
#         det (torch.Tensor): 输入的热图张量。
#
#         返回:
#         list: 调整后的关键点数组列表。
#         """
#         # 遍历每个样本
#         for batch_id, people in enumerate(ans):
#             # 遍历每个人
#             for people_id, i in enumerate(people):
#                 # 遍历每个人的每个关节
#                 for joint_id, joint in enumerate(i):
#                     # 如果关节的置信度大于 0
#                     if joint[2] > 0:
#                         # 获取关节的 y 和 x 坐标
#                         y, x = joint[0:2]
#                         # 将坐标转换为整数
#                         xx, yy = int(x), int(y)
#                         # 获取当前样本的当前关节的热图
#                         tmp = det[batch_id][joint_id]
#                         # 如果当前关节的热图在 y+1 位置的值大于 y-1 位置的值
#                         if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
#                             # 向上微调 y 坐标
#                             y += 0.25
#                         else:
#                             # 向下微调 y 坐标
#                             y -= 0.25
#
#                         # 如果当前关节的热图在 x+1 位置的值大于 x-1 位置的值
#                         if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
#                             # 向右微调 x 坐标
#                             x += 0.25
#                         else:
#                             # 向左微调 x 坐标
#                             x -= 0.25
#                         # 更新关节的坐标
#                         ans[batch_id][people_id, joint_id, 0:2] = (y + 0.5, x + 0.5)
#         return ans
#
#     def refine(self, det, tag, keypoints):
#         """
#         给定初始的关键点预测，识别缺失的关节。
#
#         参数:
#         det (numpy.ndarray): 热图数组，形状为 (关节数, 高度, 宽度)。
#         tag (numpy.ndarray): 标签数组，形状为 (关节数, 高度, 宽度) 或 (关节数, 高度, 宽度, 标签维度)。
#         keypoints (numpy.ndarray): 关键点数组，形状为 (关节数, 4)，最后一维为 (x, y, 检测分数, 标签分数)。
#
#         返回:
#         numpy.ndarray: 细化后的关键点数组。
#         """
#         # 如果标签数组的维度为 3，则在最后一维添加一个维度
#         if len(tag.shape) == 3:
#             tag = tag[:, :, :, None]
#
#         # 用于存储已检测到的关键点的标签
#         tags = []
#         # 遍历每个关键点
#         for i in range(keypoints.shape[0]):
#             # 如果关键点的置信度大于 0
#             if keypoints[i, 2] > 0:
#                 # 获取关键点的 x 和 y 坐标
#                 x, y = keypoints[i][:2].astype(np.int32)
#                 # 将关键点的标签添加到 tags 列表中
#                 tags.append(tag[i, y, x])
#
#         # 计算已检测到的关键点的平均标签
#         prev_tag = np.mean(tags, axis=0)
#         # 用于存储细化后的关键点信息
#         ans = []
#
#         # 遍历每个关键点
#         for i in range(keypoints.shape[0]):
#             # 获取当前关节的热图
#             tmp = det[i, :, :]
#             # 计算当前关节的标签与已检测到的关键点的平均标签之间的差异
#             tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
#             # 将热图减去差异
#             tmp2 = tmp - np.round(tt)
#
#             # 找出 tmp2 中最大值的位置
#             y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
#             xx = x
#             yy = y
#             # 获取最大值位置的检测置信度
#             val = tmp[y, x]
#             # 对坐标进行微调
#             x += 0.5
#             y += 0.5
#
#             # 进一步微调坐标
#             if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
#                 x += 0.25
#             else:
#                 x -= 0.25
#
#             if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
#                 y += 0.25
#             else:
#                 y -= 0.25
#
#             # 将细化后的关键点信息添加到 ans 列表中
#             ans.append((x, y, val))
#         # 将 ans 列表转换为 numpy 数组
#         ans = np.array(ans)
#
#         # 如果 ans 不为空
#         if ans is not None:
#             # 遍历每个关键点
#             for i in range(det.shape[0]):
#                 # 如果细化后的关键点的置信度大于 0 且原始关键点的置信度为 0
#                 if ans[i, 2] > 0 and keypoints[i, 2] == 0:
#                     # 更新原始关键点的坐标和置信度
#                     keypoints[i, :2] = ans[i, :2]
#                     keypoints[i, 2] = ans[i, 2]
#
#         return keypoints
#
#     def parse(self, det, tag, adjust=True, refine=True):
#         """
#         解析热图和标签图，提取关键点信息。
#
#         参数:
#         det (torch.Tensor): 输入的热图张量。
#         tag (torch.Tensor): 输入的标签张量。
#         adjust (bool): 是否对关键点的位置进行微调，默认为 True。
#         refine (bool): 是否对关键点进行细化，默认为 True。
#
#         返回:
#         tuple: 包含分组后的关键点列表和每个人的关键点平均置信度列表的元组。
#         """
#         # 从热图和标签图中提取前 max_num_people 个候选点，并进行匹配
#         ans = self.match(**self.top_k(det, tag))
#
#         # 如果需要调整关键点的位置
#         if adjust:
#             # 调用 adjust 方法调整关键点的位置
#             ans = self.adjust(ans, det)
#
#         # 计算每个人的关键点的平均置信度
#         scores = [i[:, 2].mean() for i in ans[0]]
#
#         # 如果需要细化关键点的位置
#         if refine:
#             # 取出第一个样本的关键点信息
#             ans = ans[0]
#             # 遍历每个检测到的人
#             for i in range(len(ans)):
#                 # 将热图张量转换为 numpy 数组
#                 det_numpy = det[0].cpu().numpy()
#                 # 将标签张量转换为 numpy 数组
#                 tag_numpy = tag[0].cpu().numpy()
#                 # 如果每个关节没有独立的标签，则将标签数组复制为每个关节都有相同的标签
#                 if not self.tag_per_joint:
#                     tag_numpy = np.tile(
#                         tag_numpy, (self.params.num_joints, 1, 1, 1)
#                     )
#                 # 调用 refine 方法细化关键点的位置
#                 ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
#             # 将细化后的关键点信息重新封装为列表
#             ans = [ans]
#
#         return ans, scores
# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Some code is from https://github.com/princeton-vl/pose-ae-train/blob/454d4ba113bbb9775d4dc259ef5e6c07c2ceed54/utils/group.py
# # Written by Bin Xiao (leoxiaobin@gmail.com)
# # Modified by Bowen Cheng (bcheng9@illinois.edu)
# # ------------------------------------------------------------------------------
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from munkres import Munkres
import numpy as np
import torch


def py_max_match(scores):
    #根据标签值的相似性，将候选关键点分配到不同人体实例
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


def match_by_tag(inp, params):
    assert isinstance(params, Params), 'params should be class Params()'
    tag_k, loc_k, val_k = inp
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]))

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate(
            (loc_k[idx], val_k[idx, :, None], tags), 1
        )
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
               and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added-num_grouped))+1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if (
                    row < num_added
                    and col < num_grouped
                    and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans


class Params(object):
    def __init__(self, cfg):
        # self.num_joints = cfg.DATASET.NUM_JOINTS
        # self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE
        self.num_joints =4
        self.max_num_people = 5
        # self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD
        # self.tag_threshold = cfg.TEST.TAG_THRESHOLD
        self.detection_threshold = 0.1
        self.tag_threshold = 1
        self.use_detection_val = cfg.TEST.USE_DETECTION_VAL
        self.ignore_too_much = cfg.TEST.IGNORE_TOO_MUCH

        if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
            self.num_joints -= 1

        if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
            self.joint_order = [
                i-1 for i in [5, 1, 2, 3, 4]
            ]
        else:
            self.joint_order = [
                i-1 for i in [1, 2, 3, 4]
            ]


class HeatmapParser(object):
    def __init__(self, cfg):
        self.params = Params(cfg)
        self.tag_per_joint = cfg.MODEL.TAG_PER_JOINT
        self.pool = torch.nn.MaxPool2d(
            cfg.TEST.NMS_KERNEL, 1, cfg.TEST.NMS_PADDING
        )

    def nms(self, det):
        #抑制重叠的候选点，保留局部极大值点。
        maxm = self.pool(det)  # 最大池化
        maxm = torch.eq(maxm, det).float()  # 保留极大值点
        det = det * maxm  # 过滤非极大值
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x: match_by_tag(x, self.params)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k(self, det, tag):
        #从热图中提取每个位置置信度最高的前 max_num_people 个候选点。
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)

        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)
        val_k, ind = det.topk(self.params.max_num_people, dim=2)

        tag = tag.view(tag.size(0), tag.size(1), w*h, -1)
        if not self.tag_per_joint:
            tag = tag.expand(-1, self.params.num_joints, -1, -1)

        tag_k = torch.stack(
            [
                torch.gather(tag[:, :, :, i], 2, ind)
                for i in range(tag.size(3))
            ],
            dim=3
        )

        x = ind % w
        y = (ind / w).long()

        ind_k = torch.stack((x, y), dim=3)

        ans = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return ans

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def refine(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return:
        """
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def parse(self, det, tag, adjust=True, refine=True):
        #提取候选关键点及其标签值
        ans = self.match(**self.top_k(det, tag))

        if adjust:
            ans = self.adjust(ans, det)

        scores = [i[:, 2].mean() for i in ans[0]]

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                det_numpy = det[0].cpu().numpy()
                tag_numpy = tag[0].cpu().numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(
                        tag_numpy, (self.params.num_joints, 1, 1, 1)
                    )
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
            ans = [ans]

        return ans, scores
