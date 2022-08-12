import torch

from net.NetModel_aspp import Net_aspp
from torchvision.utils import save_image
from PIL import Image
import numpy as np


cuda = True


def get_clear_img_infer(net_clear, img):
    image = img.convert('RGB')
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    images = [image]
    images = torch.from_numpy(np.asarray(images))

    if cuda:
        input_img = images.cuda()
    else:
        input_img = images
    with torch.no_grad():
        output_image = net_clear(input_img)  # 输出tensor
        save_image(output_image, 'temp/' + 'temp.png')  # 直接保存张量图片，自动转换
'''

def get_clear_img(net, img_path):

    image = Image.open(img_path).convert('RGB')
    get_clear_img_infer(net, image)
'''

if __name__ == "__main__":
    img_class = 'rainstreakdrop'

    model_path = ''
    if img_class == 'gt':
        model_path = ''

    if img_class == 'fog':
        model_path = 'workdirs/aspp_fog_ssim.pth'

    if img_class == 'raindrop':
        model_path = 'workdirs/aspp.pth'

    if img_class == 'rainstreak':
        model_path = 'workdirs/aspp_derain.pth'

    if img_class == 'rainstreakdrop':
        model_path = 'workdirs/aspp_2d.pth'


    if model_path == '':
        #cv2.imwrite('temp/temp.jpg', img)
        1
        print('gt')
    else:
        if cuda:
            net = Net_aspp().cuda()
            net.load_state_dict(torch.load(model_path))  # 加载训练好的模型参数
        else:
            net = Net_aspp()
            net.load_state_dict(torch.load(model_path, map_location='cpu'))  # 加载训练好的模型参数

        img_path = 'temp/2_rainstreakdrop.png'
        get_clear_img(net, img_path)


