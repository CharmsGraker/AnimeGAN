import sys
import os
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_dir) + os.path.sep + '.')

sys.path.append(root_path)


import datetime

import cv2 as cv
import os

from tools.data_sampler import sampleFile
from tools.error import quitError, DeleteImgError
from tools.utils import check_folder, Parallel


class imgValidator(object):
    """
        用来筛选那些合格的资料，不合格的全部剔除
    """
    def __init__(self, sample_name,
                 sample_dir='sampling',
                 dataset_dir='dataset',
                 style_dir="style"):
        """

        :param sample_name: 全称。比如Perfect Blue,
        :param sample_dir:
        :param dataset_dir:
        """
        assert sample_dir is not None, "sample folder can't be empty."
        cur_path = os.getcwd()
        sample_abs_dir = os.path.join(cur_path, sample_dir, sample_name)
        assert os.path.exists(sample_abs_dir),  "sample folder can't be found."
        self.sample_abs_dir = sample_abs_dir
        print('absolute path: ',self.sample_abs_dir)
        self.sample_dir = sample_dir
        self.sample_name = sample_name

        self.dataset_dir = dataset_dir

        self.style_dir = style_dir


    def validate(self):
        # sample_path = os.path.join(self.sample_dir, self.sample_name)

        # https://zhuanlan.zhihu.com/p/149824829
        # 默认从上往下，DFS遍历
        data_cnt = 0
        img_cnt = 0
        window = "validator on {}".format(self.sample_name)

        print('start validate image quality')
        walkfolder = self.sample_abs_dir
        print('walk in ', walkfolder)
        try:
            for cur_dir, dirs, files in os.walk(walkfolder):
                # print(files)
                for file in files:
                    # print(file)
                    if file.split('.')[-1] in ['jpg', 'png', 'jpeg']:

                        img_name = datetime.datetime.now().strftime("%d_%H-%M-%S") + "_{}".format(img_cnt)+'.jpg'
                        # print(img_name)
                        img_cnt += 1
                        img_path = os.path.join(cur_dir, file)
                        img = cv.imread(img_path)
                        if img is None:
                            raise DeleteImgError()
                        cv.imshow(window, img)

                        while True:
                            # 回车键保留当前资料
                            if cv.waitKey(1000) == ord('\r'):
                                data_cnt += 1
                                self.saveImg2dataset(img, img_name)
                                break

                            if cv.waitKey(1000) == ord(' '):
                                img_dir = os.path.join(walkfolder, cur_dir, file)
                                self.dropImgwithDir(img_dir)
                                break

                            # 点击q退出
                            if cv.waitKey(1000) == ord('q'):
                                quit = True
                                raise quitError()

                            # 点击窗口关闭按钮退出程序
                            if cv.getWindowProperty(window, cv.WND_PROP_AUTOSIZE) < 1:
                                raise quitError()
            cv.destroyWindow(window)

            msg = "sample validate name: {}\n" \
                  "dataset image count: {}\n" \
                  "sample count: {}\n".format(self.sample_name, data_cnt, img_cnt)
            print(msg)

        except quitError:
            cv.destroyWindow(window)
            print('interupt by user')
        except DeleteImgError:
            cv.destroyWindow(window)
            self.init()

            t = threading.Thread(target=self.validate)
            t.start()

    @Parallel
    def saveImg2dataset(self, img, img_name):
        """
        将当前图片保存到dataset,为了不占用当前资源，开一个后台线程去保存
        :param img:
        :return:
        """
        # def work(img, img_name):
        set_save_dir = os.path.join(self.dataset_dir, self.sample_name, self.style_dir)
        check_folder(set_save_dir)

        cur_img_save_dir = os.path.join(set_save_dir, img_name)

        cv.imwrite(cur_img_save_dir, img)
        print('success save image {}'.format(img_name))
        # t = threading.Thread(target=work,args=(img,img_name))
        # t.start()

    @Parallel
    def dropImgwithDir(self, img_dir):
        """
        为了不占用当前资源，开一个后台线程去保存
        :param img_dir:
        :return:
        """

        if os.path.exists(img_dir):
            os.remove(img_dir)
            print('drop a image in path {}'.format(img_dir))

    def init(self):
        self.__init__(sample_name=self.sample_name)
        print('reinit done.')


def main():
    if not sampleFile:
        validate_on = "Perfect Blue"
    else:
        validate_on = sampleFile[0]
    img_val = imgValidator(sample_name=validate_on)
    img_val.validate()


if __name__ == '__main__':
    main()

