import json

# 假设你有一个包含图片标注信息的JSON文件
with open('E:\coco\annotations\person_keypoints_train2017.json', 'r') as f:
    annotations = json.load(f)

# 存储每张纸的面积
areas = []

# 遍历每张文档的标注
for annotation in annotations['annotations']:
    # 假设每个文档有一个bbox字段，格式是 [x_min, y_min, width, height]
    x_min, y_min, width, height = annotation['bbox']

    # 计算面积
    area = width * height
    areas.append(area)

# 找出最小的纸张面积
min_area = min(areas)
print(f"最小纸张面积是: {min_area} 像素²")
