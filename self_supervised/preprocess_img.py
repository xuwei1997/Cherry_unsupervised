from PIL import Image
import numpy as np
import os


def preprocess_input(image):
    image = image / 127.5 - 1
    return image


def one_img(img_path):
    input_shape = [576, 1024]
    # input_shape = [288, 512]
    image = Image.open(img_path)
    h, w = input_shape
    image = image.resize((w, h), Image.BICUBIC)
    # image.show()
    # print(image.size)
    jpg = preprocess_input(np.array(image, np.float32))
    # print(jpg.shape)

    return jpg


def double_img(dou_img):
    path = '/home/dell/out'
    # path = 'E:\charry\out'
    # path='/mnt/hdd/cherry2021/out'
    # print(dou_img)
    img0 = os.path.join(path, dou_img[0])
    img1 = os.path.join(path, dou_img[1])
    jpg0 = one_img(img0)
    jpg1 = one_img(img1)
    img_out = np.stack((jpg0, jpg1))
    # print(img_out.shape)
    return img_out


if __name__ == '__main__':
    # img_path='E:\charry\out\E88569964p1t09i0d2021-01-06_09.jpg'
    # j=one_img(img_path)
    img_path = 'E:\charry\out'
    double_img(['E88570011p4t15i2d2021-03-14_15.jpg', 'E88570046p5t15i2d2021-03-01_15.jpg'])
