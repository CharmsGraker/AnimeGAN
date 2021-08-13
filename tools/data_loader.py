import os
import tensorflow as tf
import cv2, random
import numpy as np


class ImageGenerator(object):
    """
        有组织的从目录取出资料
        这里面未涉及图片裁切工作

    """
    def __init__(self, image_dir, size, batch_size, data_mean, num_cpus=16):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.num_cpus = num_cpus
        self.size = size
        self.batch_size = batch_size
        self.data_mean = np.asarray(data_mean).astype(np.float32)

    def get_image_paths_train(self, image_dir):
        """
        读取训练集资料的path
        :param image_dir:
        :return:
        """
        image_dir = os.path.join(image_dir)

        paths = []

        """
            path: 一个文件条目
        """
        for path in os.listdir(image_dir):
            # Check extensions of filename
            # 只处理图片
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            paths.append(path_full)

        return paths

    def read_image(self, img_path1):

        if 'style' in img_path1.decode() or 'smooth' in img_path1.decode():
            # color image1
            image1 = cv2.imread(img_path1.decode()).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            # image1 现在是,        data mean是
            #               0 1 2           0 1 2
            #               R G B,          B G R
            image1[:, :, 0] += self.data_mean[2]
            image1[:, :, 1] += self.data_mean[1]
            image1[:, :, 2] += self.data_mean[0]

            # gray image2
            image2 = cv2.imread(img_path1.decode(), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            image2 = np.asarray([image2, image2, image2])
            # 看样子cv读取图片默认读成BGR
            # 如果原来的维度标号记为0,1,2；那么现在维度排列变为1,2,0
            # 也就是从BGR变成RGB
            image2 = np.transpose(image2, (1, 2, 0))

        else:
            # 否则表明这是训练资料，其灰度图直接设为0
            # color image1
            image1 = cv2.imread(img_path1.decode()).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            image2 = np.zeros(image1.shape).astype(np.float32)

        return image1, image2

    def load_image(self, img1):
        """

        :param img1: 将图片颜色分量[0,255]映射为[-1 , 1]，供训练使用
        :return:
        """
        image1, image2 = self.read_image(img1)
        # 缩放至-1 ~ 1
        processing_image1 = image1 / 127.5 - 1.0
        processing_image2 = image2 / 127.5 - 1.0
        return (processing_image1, processing_image2)

    def load_images(self):
        """
            设置dataset的迭代器，以后通过tf.sess来从数据集取出图像数据
        :return: 取出dataset的下一个元素，color，gray
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        # Repeat indefinitely
        dataset = dataset.repeat()

        # Unform shuffle
        dataset = dataset.shuffle(buffer_size=len(self.paths))

        # Map path to image 
        dataset = dataset.map(lambda img: tf.py_func(
            self.load_image, [img], [tf.float32, tf.float32]),
                              self.num_cpus)

        dataset = dataset.batch(self.batch_size)

        # 得到迭代器取出的第一个元素，之后就通过tf.sess获取了
        img1, img2 = dataset.make_one_shot_iterator().get_next()

        return img1, img2
