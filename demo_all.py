import os
import torch
from PIL import Image
import cv2
import time
import numpy as np

from get_class import get_img_class_infer
from net.network_class import Network_class

from get_clear import get_clear_img_infer
from net.NetModel_aspp import Net_aspp

from yolo import YOLO

cuda = True  # 修改时需要同时对yolo.py文件中的cuda做出同样修改

class_list = ['gt', 'fog', 'raindrop', 'rainstreak', 'rainstreakdrop']
mode = 'img'


def get_result_img_label(img, net_class, net_clear_fog, net_clear_raindrop, net_clear_rainstreak,
                         model_clear_rainstreakdrop):
    max_idx = get_img_class_infer(net_class, img)
    img_class = class_list[max_idx]
    # img_class = 'gt'
    print(img_class)
    # net_clear = ''
    if img_class == 'gt':
        img.save('temp/temp.png')

    elif img_class == 'fog':
        # net_clear = net_clear_fog
        get_clear_img_infer(net_clear_fog, img)

    elif img_class == 'raindrop':
        # net_clear = net_clear_raindrop
        get_clear_img_infer(net_clear_raindrop, img)

    elif img_class == 'rainstreak':
        # net_clear = model_path_rainstreak
        get_clear_img_infer(net_clear_rainstreak, img)

    elif img_class == 'rainstreakdrop':
        # net_clear = model_path_rainstreakdrop
        get_clear_img_infer(model_clear_rainstreakdrop, img)

    # if img_class == 'gt':
    #     img.save('temp/temp.png')
    # else:
    #     get_clear_img_infer(net_clear, img)

    img_obj = Image.open('temp/temp.png')

    r_image, label_informetion = yolo.detect_image(img_obj, crop=False, count=False)

    return r_image, label_informetion


if __name__ == "__main__":

    class_num = len(class_list)
    model_path_class = 'checkpoint/classification.pth'

    ###### 加载分类模型 ######
    net_class = Network_class(class_num=class_num)
    if cuda:
        net_class = net_class.cuda()
    if os.path.exists(model_path_class):
        net_class.load_state_dict(torch.load(model_path_class))  # 加载训练过的模型
    else:
        print("error！ 无法打开模型 模型路径：", model_path_class)
        exit()

    ###### 加载去雨模型 ######
    model_path_fog = 'checkpoint/fog_aspp_3_1pth'
    model_path_raindrop = 'checkpoint/aspp.pth'
    model_path_rainstreak = 'checkpoint/rainstreak_aspp_3pth'
    model_path_rainstreakdrop = 'checkpoint/2r_aspp_3pth'

    net_clear_fog = Net_aspp()
    net_clear_raindrop = Net_aspp()
    net_clear_rainstreak = Net_aspp()
    net_clear_rainstreakdrop = Net_aspp()

    if cuda:
        net_clear_fog = net_clear_fog.cuda()
        net_clear_raindrop = net_clear_raindrop.cuda()
        net_clear_rainstreak = net_clear_rainstreak.cuda()
        net_clear_rainstreakdrop = net_clear_rainstreakdrop.cuda()
    if os.path.exists(model_path_class):
        net_clear_fog.load_state_dict(torch.load(model_path_fog))  # 加载训练过的模型
        net_clear_raindrop.load_state_dict(torch.load(model_path_raindrop))  # 加载训练过的模型
        net_clear_rainstreak.load_state_dict(torch.load(model_path_rainstreak))  # 加载训练过的模型
        net_clear_rainstreakdrop.load_state_dict(torch.load(model_path_rainstreakdrop))  # 加载训练过的模型
    else:
        print("error！ 无法打开模型 模型路径：", model_path_fog)
        print("error！ 无法打开模型 模型路径：", model_path_raindrop)
        print("error！ 无法打开模型 模型路径：", model_path_rainstreak)
        print("error！ 无法打开模型 模型路径：", model_path_rainstreakdrop)
        exit()
    ###### 加载检测模型 ######
    yolo = YOLO()

    if mode == 'img':
        img_path = './img/524_rainstreakdrop.png'
        img = Image.open(img_path).convert('RGB')
        img_out, label = get_result_img_label(img, net_class, net_clear_fog, net_clear_raindrop, net_clear_rainstreak,
                                              net_clear_rainstreakdrop)

        print(label)
        img_out.show()
        img_out.save('img_out/'+img_path.split('/')[-1])

    if mode == 'video':
        fps = 1.0
        # use camera or use video
        capture = cv2.VideoCapture(0)
        # capture = cv2.VideoCapture('data/test/video/testb1.mp4')
        fps_origin = capture.get(cv2.CAP_PROP_FPS)
        frame_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(frame_h,frame_w)
        while (True):
            t1 = time.time()
            # read one frame
            ref, frame = capture.read()
            frame = cv2.resize(frame, (320, 240))

            cv2.imshow('o', frame)

            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            img_out, label = get_result_img_label(frame, net_class, net_clear_fog, net_clear_raindrop,
                                                  net_clear_rainstreak,
                                                  net_clear_rainstreakdrop)

            img_out = cv2.cvtColor(np.asarray(img_out), cv2.COLOR_RGB2BGR)
            cv2.imshow('out', img_out)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print(fps)
            # frame_out = cv2.putText(frame_out, 'fps=%.2f' % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            c = cv2.waitKey(1) & 0xff

            if c == 27:
                capture.release()
                break
