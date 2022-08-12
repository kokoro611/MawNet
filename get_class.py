import torch

import os
from net.network_class import Network_class
from tqdm import tqdm
import numpy as np

from PIL import Image

cuda = True
class_list = ['gt', 'fog', 'raindrop', 'rainstreak', 'rainstreakdrop']


def img_crop_center(img, w, h):
    img_width, img_height = img.size
    left, right = (img_width - w) / 2, (img_width + w) / 2
    top, bottom = (img_height - h) / 2, (img_height + h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    return img.crop((left, top, right, bottom))


def get_img_class_infer(net_class, image):
    if image.size[0] < 224:
        image = image.resize((224, int(image.size[1] * (224 / image.size[0]))))
    if image.size[1] < 224:
        image = image.resize((int(image.size[0] * (224 / image.size[1])), 224))

    cropped = img_crop_center(image, 224, 224)

    image = np.array(cropped, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    images = [image]
    images = torch.from_numpy(np.asarray(images))

    if cuda:
        input = images.cuda()
    else:
        input = images
    with torch.no_grad():
        output = net_class(input)  # 输出tensor
    output = output.cpu()[0]
    output = list(output)
    max_value = max(output)  # 求列表最大值
    max_idx = output.index(max_value)
    return max_idx


def get_img_class(net, image):
    #img = Image.open(image_path).convert('RGB')
    max_idx = get_img_class_infer(net, image)
    label = class_list[max_idx]
    return label


if __name__ == "__main__":

    class_num = len(class_list)
    model_path = 'checkpoint/classification.pth'

    net = Network_class(class_num=class_num)
    if cuda:
        net = net.cuda()

    if os.path.exists(model_path):  # 判断模型有没有提前训练过
        print("开始测试！")
        net.load_state_dict(torch.load(model_path))  # 加载训练过的模型
    else:
        print("error！ 无法打开模型 模型路径：", model_path)
        exit()

    img_path = './data/test/2_fog.png'
    label = get_img_class(net, img_path)
    print(label)
