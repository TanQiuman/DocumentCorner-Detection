import os
import json
import cv2

def resize_image_and_update_json(image_path, json_path, output_image_dir, output_json_dir, target_size=(256, 256)):
    """
    调整图片大小并更新对应的 JSON 文件
    :param image_path: 原始图片路径
    :param json_path: 原始 JSON 文件路径
    :param output_image_dir: 调整后的图片保存目录
    :param output_json_dir: 更新后的 JSON 文件保存目录
    :param target_size: 目标图片大小，默认为 (256, 256)
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图片 {image_path}，跳过处理。")
        return

    # 获取原始图片的宽度和高度
    original_height, original_width = image.shape[:2]

    # 调整图片大小
    resized_image = cv2.resize(image, target_size)

    # 计算缩放比例
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 更新点坐标
    for shape in data['shapes']:
        for i in range(len(shape['points'])):
            x, y = shape['points'][i]
            shape['points'][i] = [x * scale_x, y * scale_y]

    # 更新图片大小信息
    data['imageHeight'] = target_size[1]
    data['imageWidth'] = target_size[0]

    # 保存调整后的图片
    image_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_image_dir, image_name)
    cv2.imwrite(output_image_path, resized_image)

    # 保存更新后的 JSON 文件
    json_name = os.path.basename(json_path)
    output_json_path = os.path.join(output_json_dir, json_name)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"已处理：{image_name} 和 {json_name}")

def process_folder(image_dir, json_dir, output_image_dir, output_json_dir, target_size=(256, 256)):
    """
    处理整个文件夹中的图片和 JSON 文件
    :param image_dir: 原始图片文件夹
    :param json_dir: 原始 JSON 文件夹
    :param output_image_dir: 调整后的图片保存目录
    :param output_json_dir: 更新后的 JSON 文件保存目录
    :param target_size: 目标图片大小，默认为 (256, 256)
    """
    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    # 遍历图片文件夹
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 构建图片和 JSON 文件路径
        image_path = os.path.join(image_dir, image_name)
        json_name = os.path.splitext(image_name)[0] + '.json'
        json_path = os.path.join(json_dir, json_name)

        if not os.path.exists(json_path):
            print(f"警告：{json_name} 不存在，跳过处理。")
            continue

        # 处理图片和 JSON 文件
        resize_image_and_update_json(image_path, json_path, output_image_dir, output_json_dir, target_size)

if __name__ == "__main__":
    # 输入文件夹路径
    image_dir = r"E:\coco\images\newpaperdataset\images"  # 原始图片文件夹
    json_dir = r"E:\coco\images\newpaperdataset\annotations"  # 原始 JSON 文件夹

    # 输出文件夹路径
    output_image_dir = r"E:\coco\images\newpaperdataset\images256"  # 调整后的图片保存目录
    output_json_dir = r"E:\coco\images\newpaperdataset\annotations256"  # 更新后的 JSON 文件保存目录

    # 处理文件夹
    process_folder(image_dir, json_dir, output_image_dir, output_json_dir)