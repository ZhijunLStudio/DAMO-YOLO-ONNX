import xml.etree.ElementTree as ET
import os
import json
import collections
import random
import shutil


category_set = ['ship']

# 设置随机数种子，可以是任意整数
random_seed = 42

# 设置随机数种子
random.seed(random_seed)


coco_train = dict()
coco_train['images'] = []
coco_train['type'] = 'instances'
coco_train['annotations'] = []
coco_train['categories'] = []

coco_val = dict()
coco_val['images'] = []
coco_val['type'] = 'instances'
coco_val['annotations'] = []
coco_val['categories'] = []

# category_set = dict()
image_set = set()
train_image_id = 1
val_image_id = 200000  # Assuming you have less than 200000 images
category_item_id = 1
annotation_id = 1


def split_list_by_ratio(input_list, ratio=0.8):
    # 计算切分的索引位置
    split_index = int(len(input_list) * ratio)
    # 随机打乱列表
    random.shuffle(input_list)
    # 划分为两个列表并返回
    return input_list[:split_index], input_list[split_index:]

def addCatItem(name):
    '''
    增加json格式中的categories部分
    '''
    global category_item_id
    category_item = collections.OrderedDict()
    category_item['supercategory'] = 'none'
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco_train['categories'].append(category_item)
    coco_val['categories'].append(category_item)
    category_item_id += 1


def addImgItem(file_name, size, img_suffixes, is_train):
    global train_image_id  # 声明变量为全局变量
    global val_image_id  # 声明变量为全局变量
    # global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    # image_item = dict()    #按照一定的顺序，这里采用collections.OrderedDict()
    image_item = collections.OrderedDict()
    jpg_name = os.path.splitext(file_name)[0] + img_suffixes
    image_item['file_name'] = jpg_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    # image_item['id'] = image_id
    # coco['images'].append(image_item)
    if is_train:
        image_item['id'] = train_image_id
        coco_train['images'].append(image_item)
        image_id = train_image_id
        train_image_id += 1
    else:
        image_item['id'] = val_image_id
        coco_val['images'].append(image_item)
        image_id = val_image_id
        val_image_id += 1
    image_set.add(jpg_name)
    image_id = image_id + 1
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox, is_train):
    global annotation_id
    # annotation_item = dict()
    annotation_item = collections.OrderedDict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_item['id'] = annotation_id
    annotation_item['ignore'] = 0
    annotation_id += 1
    # coco['annotations'].append(annotation_item)
    if is_train:
        coco_train['annotations'].append(annotation_item)
    else:
        coco_val['annotations'].append(annotation_item)

def parseXmlFiles(xml_path, xmllist, img_suffixes, is_train):
    for f in xmllist:
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()  # 抓根结点元素

        if root.tag != 'annotation':  # 根节点标签
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            # elem.tag, elem.attrib，elem.text
            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size, img_suffixes, is_train)  # 图片信息
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    # if object_name not in category_set:
                    #    current_category_id = addCatItem(object_name)
                    # else:
                    # current_category_id = category_set[object_name]
                    current_category_id = category_set.index(object_name) + 1  # index默认从0开始,但是json文件是从1开始，所以+1
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print(
                        'add annotation with {},{},{},{}'.format(object_name, current_image_id - 1, current_category_id,
                                                                 bbox))
                    addAnnoItem(object_name, current_image_id - 1, current_category_id, bbox, is_train)



def copy_img(img_path, file_list, img_suffixes, new_folder):
    # global train_image_id  # 将train_image_id声明为全局变量
    # global val_image_id  # 将val_image_id声明为全局变量

    parent_directory = os.path.dirname(img_path)
    dest_folder = os.path.join(parent_directory, new_folder)
    # 创建目标文件夹
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for each_file in file_list:
        file_prefix = os.path.splitext(each_file)[0]
        old_img_path = os.path.join(img_path, file_prefix + img_suffixes)
        new_img_path = os.path.join(dest_folder, file_prefix + img_suffixes)
        shutil.copy(old_img_path, new_img_path)
        # print(f'已拷贝图片到{new_img_path}')

        # 更新image_id
        # if new_folder == 'train':
        #     train_image_id += 1
        # else:
        #     val_image_id += 1



def check_image_folder_suffix(folder_path):
    # 获取文件夹中所有文件的后缀名，并将它们放入一个集合(set)中
    file_suffixes = set()
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            _, file_suffix = os.path.splitext(file_name)
            file_suffixes.add(file_suffix)

    # 检查集合中后缀名的数量，如果数量为1，则所有图片都是同一个后缀，返回后缀名，否则报错
    assert len(file_suffixes) == 1, "图片文件夹中的后缀名不统一"
    return file_suffixes.pop()



if __name__ == '__main__':
    # 存放img和xml的文件夹
    img_path = 'data/images'
    xml_path = 'data/Annotations'
    # 确保img文件夹中只有一种格式
    img_suffixes = check_image_folder_suffix(img_path)
    annotation_folder = os.path.join('data', 'annotations')
    os.makedirs(annotation_folder, exist_ok=True)
    # 保存生成的coco格式的json路径
    train_json_file = os.path.join(annotation_folder, 'instances_train2017.json')
    val_json_file = os.path.join(annotation_folder, 'instances_val2017.json')
    # 添加categories部分
    for categoryname in category_set:
        addCatItem(categoryname)
    # 获取所有的XML文件列表
    xmllist = os.listdir(xml_path)
    # 按8:2的随机比例划分为两个列表
    train_list, val_list = split_list_by_ratio(xmllist, ratio=0.8)
    print(train_list)
    print('--------------------')
    print(val_list)
    # 拷贝图片到新的文件夹
    copy_img(img_path, train_list, img_suffixes, 'train2017')
    copy_img(img_path, val_list, img_suffixes, 'val2017')
    parseXmlFiles(xml_path, train_list, img_suffixes, True)
    parseXmlFiles(xml_path, val_list, img_suffixes, False)
    json.dump(coco_train, open(train_json_file, 'w'))
    json.dump(coco_val, open(val_json_file, 'w'))
