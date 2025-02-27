import glob
import json
import cv2
import os
import numpy as np
import re
import shutil


# 去掉文件名中的中文字符
def remove_chinese_from_filename(filename):
    # 使用正则表达式去掉文件名中的中文字符
    return re.sub(r'[\u4e00-\u9fff]+', '', filename)


def drawPointsAndRect(points, img, imgpath):
    dict = {0: "LT", 1: "RT", 2: "RD", 3: "LD"}
    for k in range(len(points)):
        # 绘制点的坐标
        for i in range(0, 7, 2):
            x, y = int(points[k][i]), int(points[k][i + 1])
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(img, dict[i // 2], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    cv2.namedWindow(imgpath, cv2.WINDOW_NORMAL)
    cv2.moveWindow(imgpath, 100, 100)  # 将窗口移动到坐标 (100, 100)
    cv2.resizeWindow(imgpath, 800, 600)  # 设置窗口大小为 800x600 像素
    cv2.imshow(imgpath, img)
    # 等待用户按下任意键，这将保持图像显示，直到用户按下键盘上的任意键
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()


def process_file(img_path, is_visible):
    points = []

    # 去掉文件名中的中文字符
    img_name = os.path.basename(img_path)
    new_img_name = remove_chinese_from_filename(img_name)

    # 获取对应的 JSON 文件路径
    label_path = img_path.replace(".jpg", ".json")

    # 重命名图片文件
    new_img_path = os.path.join(os.path.dirname(img_path), new_img_name)
    if img_path != new_img_path:
        shutil.move(img_path, new_img_path)
        img_path = new_img_path  # 更新 img_path 为新的路径

    # 重命名对应的 JSON 文件
    new_label_path = os.path.join(os.path.dirname(label_path), new_img_name.replace(".jpg", ".json"))
    if label_path != new_label_path:
        shutil.move(label_path, new_label_path)
        label_path = new_label_path  # 更新 label_path 为新的路径

    # 读取 JSON 文件
    with open(label_path, 'r', encoding='utf-8') as json_file:
        label = json.load(json_file)
        label = label['shapes']
        paperNums = len(label)
        for i in range(len(label)):
            temp = label[i]['points']
            point = []
            for j in range(len(temp)):
                point.append(int(temp[j][0]))
                point.append(int(temp[j][1]))
            points.append(point)

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片：{img_path}")
        return

    print(f"处理图片：{img_path}")

    # 显示图像
    if is_visible:
        drawPointsAndRect(points, img, new_img_name)


def traverse_directory_with_glob(directory_path, file_extension, is_visible):
    # 使用 glob 模块筛选目录下的特定格式文件
    file_pattern = os.path.join(directory_path, f"*.{file_extension}")
    for file_path in glob.glob(file_pattern):
        process_file(file_path, is_visible)


if __name__ == "__main__":
    # 指定目录路径和文件格式（例如：jpg）
    target_directory = r"E:\coco\images\newpaperdataset\images256"  # 修改为你的文件夹路径
    file_extension = "jpg"
    traverse_directory_with_glob(target_directory, file_extension, True)
