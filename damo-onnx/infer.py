import cv2
import os
import copy
from damoyolo.damoyolo_onnx import DAMOYOLO
import torch

def main():
    # 指定参数
    model_path = 'damoyolo/model/damoyolo_tinynasL35_M.onnx'
    score_th = 0.4
    nms_th = 0.85
    coco_classes = get_coco_classes()

    # 指定输入图片文件夹和输出图片文件夹
    input_folder = '/home/lzj/ssd2t/01.my_algo/damo-yolo/datasets/data/val2017'
    output_folder = 'output_images_folder'

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # 初始化模型
    model = DAMOYOLO(model_path)
    print(f"Model loaded: {model_path}")
    # 遍历输入文件夹中的图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # 进行推理
            bboxes, scores, class_ids = model(image, nms_th=nms_th)

            # 绘制结果并保存图片
            result_image = draw_debug(image, score_th, bboxes, scores, class_ids, coco_classes)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_image)
            print(f"Output saved: {output_path}")

def draw_debug(image, score_th, bboxes, scores, class_ids, coco_classes):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        # 绘制边界框
        debug_image = cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # 显示类别和分数
        score = '%.2f' % score
        text = '%s:%s' % (str(coco_classes[int(class_id)]), score)
        debug_image = cv2.putText(debug_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)

    return debug_image

def get_coco_classes():
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')
    return coco_classes

if __name__ == '__main__':
    main()
