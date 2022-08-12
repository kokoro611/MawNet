import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

yolo = YOLO()


def predict_img(img):
    r_image, label_informetion= yolo.detect_image(img, crop=False, count=False)
    return r_image, label_informetion


if __name__ == "__main__":
    img_path = 'temp/temp.png'
    image = Image.open(img_path)
    r_image, label_informetion = predict_img(image)
    print(label_informetion)
    r_image.show()

