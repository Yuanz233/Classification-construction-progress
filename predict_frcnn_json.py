#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import os
import numpy as np
from PIL import Image
import json
from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    test_interval   = 100
    root  = "./datasets2"
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    train_images_path = []  # 存储训练集的所有图片路径
    val_images_path = []  # 存储验证集的所有图片路径
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    json_dict = dict()
    count = 0
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                    if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        for tmp_path in images:
            tmp_image = Image.open(tmp_path)
            tmp_info = frcnn.cal_json_str(tmp_image)
            tmp_info = np.array(tmp_info,dtype=float).tolist()
            others, pic_name = tmp_path.rsplit('\\', 1)
            json_dict[pic_name] = tmp_info
            count += 1
            if count % 10 == 0:
                print('number of images:',count)
            if type(tmp_info)==np.ndarray:
                print('ndarray:',count,pic_name)
            if len(tmp_info) != 7:
                print('OD_info error! pic_name:',pic_name,' class:',cla)

    print('number of all images:',count)
    json_str = json.dumps(json_dict,indent=4)
    with open('OD_info.json', 'w') as json_file:
        json_file.write(json_str)

