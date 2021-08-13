# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow
# https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_dir) + os.path.sep + '.')

sys.path.append(root_path)


from tools.utils import check_folder
import numpy as np
import cv2, argparse
from glob import glob
from tqdm import tqdm


def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Paprika', help='dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()


def make_edge_smooth(dataset_name, img_size):
    """

    :param dataset_name: 默认Hayao
    :param img_size:
    :return:
    """

    # print(glob(os.pardir + '/dataset/{}/{}/*.*'.format(dataset_name, 'style')))

    check_folder(os.pardir + '/dataset/{}/{}'.format(dataset_name, 'smooth'))

    file_list = glob(os.pardir + '/dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    # 指定平滑后的图像保存位置
    save_dir = os.pardir + '/dataset/{}/smooth'.format(dataset_name)

    # 平滑时，需要参考的窗口大小
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 高斯平滑图像
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    assert file_list, "file list can not be empty."

    for f in tqdm(file_list):
        # 依次处理每个图片
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        # resize
        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        # W,H,Channel
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        # 发现边缘
        # https://blog.csdn.net/jinwanchiji/article/details/78950889
        # 100 200 是min max threshold
        edges = cv2.Canny(gray_img, 100, 200)
        #cv2.dilate()
        # 膨胀：将前景物体变大，理解成将图像断开裂缝变小（在图片上画上黑色印记，印记越来越小）
        # https://blog.csdn.net/ctrigger/article/details/99835150

        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)

        # np.where
        # https://www.cnblogs.com/massquantity/p/8908859.html
        # 返回的是满足condition的坐标，比如有三个维度，那么就会返回三个array，每个array对应了其在一个维度上的位置
        idx = np.where(dilation != 0)
        # 对于所有dilation!=0的那些像素进行操作。所以np.sum(dilation1=0)就是得到这些不合格点的个数，然后依次遍历
        for i in range(np.sum(dilation != 0)):
            # 这是指那些比较尖锐的边？
            # 分别处理三个通道
            # kernel size 范围内的gauss blur 和（沿axis=0)代替
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))

            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))

            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
        # 写入图像
        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth(args.dataset, args.img_size)


if __name__ == '__main__':
    main()
