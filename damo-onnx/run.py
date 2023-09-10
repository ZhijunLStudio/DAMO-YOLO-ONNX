import cv2
import os
from damoyolo.damoyolo_onnx import DAMOYOLO

def get_coco_classes():
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')
    return coco_classes


def main():
    # 指定参数
    model_path = 'damoyolo/model/damoyolo_tinynasL35_M.onnx'
    score_th = 0.3
    nms_th = 0.4
    coco_classes = get_coco_classes()

    # 指定输入图片文件夹和输出图片文件夹
    input_folder = '/work/data/object-detection-for-tilting-ships-test-set'
    output_folder = '/work/output/'
    # input_folder = '/home/lzj/ssd2t/01.my_algo/damo-yolo/datasets/data/val2017'
    # output_folder = 'output_images_folder'

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
            bboxes, scores, class_ids = model(image, score_th=score_th,nms_th=nms_th)

            # 生成 TXT 文件路径
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_output_path = os.path.join(output_folder, txt_filename)

            # 生成 TXT 文件内容
            with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
                for bbox, score, class_id in zip(bboxes, scores, class_ids):
                    x1, y1, x2, y2 = bbox.astype(int)
                    confidence = score
                    class_name = coco_classes[int(class_id)]
                    txt_file.write(f"{class_name} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

            print(f"Output saved: {txt_output_path}")

if __name__ == '__main__':
    main()
