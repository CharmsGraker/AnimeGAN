import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_dir) + os.path.sep + '.')

sys.path.append(root_path)

import argparse
import datetime
import threading

import numpy as np
import cv2 as cv

from tools.utils import check_folder, Parallel

"""
https://blog.csdn.net/weixin_43673589/article/details/105248883

https://www.jianshu.com/p/0733cca39148

https://blog.csdn.net/qq_44740544/article/details/106184890
"""
sampleFile = []


class videoReader(object):
    def __init__(self, video_path):
        self.video_path = video_path
        # 得到文件名
        self.name = os.path.basename(video_path)[:os.path.basename(video_path).rfind('.')]

        self.capture = cv.VideoCapture(video_path)
        self.fps = int(self.capture.get(cv.CAP_PROP_FPS))  # capture.get(5)
        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))  # capture.get(3)
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))  # capture.get(4)
        self.frame_count = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))  # int(capture.get(7))
        self.channel = 3

    def print(self):
        print('name: {},fps: {}, width: {}, height: {}, frame count: {}'.format(self.name, self.fps, self.width,
                                                                                self.height, self.frame_count))

    def timecode_to_frames(self, time_code):
        framerate = self.fps
        return sum(f * int(t) for f, t in zip((3600 * framerate, 60 * framerate, framerate, 1), time_code.split(':')))

    def frames_to_timecode(self, frames):
        framerate = self.fps
        return '{0:02d}:{1:02d}:{2:02d}:{3:02d}'.format(frames / (3600 * framerate),
                                                        frames / (60 * framerate) % 60,
                                                        frames / framerate % 60,
                                                        frames % framerate)

    def resetCapture(self):
        self.capture = cv.VideoCapture(self.video_path)

    def release(self):
        return self.capture.release()


"""
https://blog.csdn.net/michaellamer/article/details/114916170
"""


class VideoSampler(object):
    """
        一个视频采样器，只管采样，不管质量
    """

    def __init__(self, video_path,
                 start_timecode=None,
                 end_timecode=None,
                 target_size=None,
                 save_dir=r".\sampling",
                 sample_frequent=200,
                 max_sample=500):
        self.source_video = videoReader(video_path)
        assert self.source_video is not None, "video can't be empty"

        self.start_timecode = start_timecode
        self.end_timecode = end_timecode
        self.target_size = target_size
        self.save_dir = save_dir
        self.sample_frequent = sample_frequent
        self.max_sample = max_sample

        self.sample_cluster = 0

        sampleFile.append(self.source_video.name)

        self.source_video.print()

    def frameValidation(self, start_timecode=None, end_timecode=None):
        """
        为了多次采样用,单独使用时必须指明采样区间
        :param start_timecode:
        :param end_timecode:
        :return:
        """
        # 处理帧数
        if start_timecode is not None:
            sample_startframe = self.source_video.timecode_to_frames(start_timecode)
            self.sample_startframe = min([sample_startframe, self.source_video.frame_count])
        else:
            self.sample_startframe = 0
        if end_timecode is not None:
            sample_endframe = self.source_video.timecode_to_frames(end_timecode)
            self.sample_endframe = min([sample_endframe, self.source_video.frame_count])
        else:
            self.sample_endframe = self.source_video.frame_count

    def oneSampling(self,
                    start_timecode=None,
                    end_timecode=None,
                    sample_frequent=None,
                    max_sample=None,
                    sample_cluster=None):

        if self.target_size is None:
            self.target_size = [256, 256]
        target_width, target_height = self.target_size

        assert target_width <= self.source_video.width \
               and target_height <= self.source_video.height, \
            "target size is too large for this video."

        if sample_frequent is None:
            sample_frequent = self.sample_frequent
        else:
            self.sample_frequent = sample_frequent
        if max_sample is None:
            max_sample = self.max_sample
        else:
            self.max_sample = max_sample

        if sample_cluster is None:
            sample_cluster = self.sample_cluster

        source_video_capture = self.source_video.capture
        # https://blog.csdn.net/fengda2870/article/details/69389141

        self.frameValidation(start_timecode, end_timecode)

        self.source_video.capture.set(cv.CAP_PROP_POS_FRAMES, self.sample_startframe)

        # init
        cur_frame = self.sample_startframe
        sample_cnt = 0

        print('start sampling----------------')

        # 先计算目标尺寸允许采样多少张。为简单起见，就采样每帧从左到右，上下不再移动
        EachcanSample = self.source_video.width // target_width

        sample_cluster_save_dir = os.path.join(self.save_dir, self.source_video.name, "{}".format(sample_cluster))
        check_folder(sample_cluster_save_dir)

        while self.source_video.capture.isOpened():
            # 读取一帧
            # 一帧可以采样多张
            # 救命，他居然是高度在第一维度，是Height,Width,Channel
            ret, frame = self.source_video.capture.read()
            if ret is False:
                break
            for group_order in range(0, EachcanSample):
                times = group_order + 1
                # 得到一张
                target_sample = cut_frame(img=frame, target_size=self.target_size, times=times)
                cur_frame_save_dir = os.path.join(sample_cluster_save_dir,
                                                  '{}_{}_{}.jpg'
                                                  .format(datetime.datetime
                                                          .now()
                                                          .strftime('%Y-%m-%d'),
                                                          cur_frame,
                                                          group_order))
                cv.imwrite(cur_frame_save_dir, target_sample)

            cv.imshow('source video', frame)
            # print(cnt)

            cur_frame += sample_frequent
            sample_cnt += 1

            # 点击q退出
            if cv.waitKey(1) == ord('q'):
                break

            # 点击窗口关闭按钮退出程序
            if cv.getWindowProperty('source video', cv.WND_PROP_AUTOSIZE) < 1:
                break

            if cur_frame >= self.sample_endframe:
                break

            if sample_cnt == max_sample:
                break

        cv.destroyWindow("source video")

        self.source_video.release()

        msg = "sample video name: {}\nsample cluster: {}\nsample count: {}\nsample size: {},{}\n".format(
            self.source_video.name,
            self.sample_cluster,
            sample_cnt, target_width, target_height)

        log_dir = os.path.join(sample_cluster_save_dir, 'log.txt')
        log_create(log_dir, msg=msg)
        self.restSampleVar()

        print("Sampling '{}' has successfully done.sample total samples: {} frames".format(self.source_video.name,
                                                                                           sample_cnt))
        print('the log generated in path: {}'.format(sample_cluster_save_dir))

    def restSampleVar(self):
        self.sample_cluster += 1
        self.sample_startframe = None
        self.sample_endframe = None

        self.sample_frequent = 5

        if self.max_sample is None:
            self.max_sample = 500

        self.source_video.resetCapture()

    def randomSampling(self, times=5, least_frequent=100, max_sample=None):
        timestamp = [0] * times
        if max_sample is None:
            max_sample = 1000

        cur_sample_cnt = 0
        sample_cnt = 0
        for i in range(times):
            np.random.seed()
            timestamp[i] = "{}:{}:{}:{}".format(np.random.randint(0, 2),
                                                np.random.randint(0, 61),
                                                np.random.randint(0, 61),
                                                np.random.randint(0, 61))
            cur_sample_cnt = max_sample = np.random.randint(20, 100)
            sample_cnt += cur_sample_cnt
            kwargs = dict(
                start_timecode=timestamp[i],
                end_timecode=None,
                sample_frequent=np.random.randint(least_frequent, 1000),
                max_sample=cur_sample_cnt,
                sample_cluster=i)
            #
            # t = threading.Thread(target=self.sampling, args=(
            #     timestamp[i],
            #     None,
            #     np.random.randint(least_frequent, 1000),
            #     np.random.randint(300, 1000),
            #     i))
            # t = threading.Thread(target=self.sampling, kwargs=kwargs)
            # t.start()

            self.oneSampling(start_timecode=timestamp[i],
                             sample_frequent=np.random.randint(least_frequent, 4000),
                             end_timecode=None,
                             max_sample=np.random.randint(5, 10),
                             sample_cluster=i)

        print('total has sampled {}'.format(sample_cnt))


def log_create(path, msg):
    file_path = path
    file = open(file_path, 'w')
    file.write(msg)


def cut_frame(img, target_size=None, times=1, train_data=True):
    """

    :param img: 图片
    :param target_size: 二维元组。width,height
    :param train_data:
    :param times: 表示这是这张图的第几次采样
    :return:
    """
    assert img is not None, "image can't be None.Please try again."

    if target_size is None:
        target_size = [256, 256]

    crop_w, crop_h = target_size
    # print(img.shape)
    # 注意openCV 组织图像的维度是高度在前
    height, width = img.shape[0:-1]
    offset_w = int(min([crop_w * (times - 1) + width / 4, width - crop_w]))
    np.random.seed()
    offset_h = int(min([height / 4 + np.random.randint(0, height - crop_h), height - crop_h]))

    if train_data:
        return img[
               offset_h:int(offset_h + crop_h),
               offset_w: int(offset_w + crop_w)]
    else:
        return img


# workpath = os.getcwd()  # 'D:\\Gary\\Program\\GitHubProject\\AnimeGAN'

def parse_args():
    desc = "Anime to sample for train"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--collection', type=str,
                        default="PB",
                        help="sample which anime? PB for Perfect Blue,MA for Millennium Actress")
    parser.add_argument('--mode', type=str, default="random", help='random/single?')
    parser.add_argument('--frequent', type=int, default=120, help="sample frequence")
    parser.add_argument('--times', type=int, default=10, help="sample times")
    parser.add_argument('--start-timecode', type=str, default="00:00:00:00", help="start timecode")
    parser.add_argument('--max-sample', type=int, default=1000, help="max sample")

    return check_args(parser.parse_args())


def check_args(args):
    return args


movie_folder = "D:/movie"


def project(video_name):
    return os.path.join(movie_folder, video_name)


def main():
    sampleFile = []

    args = parse_args()

    collection_name = args.collection
    frequent = args.frequent
    times = args.times
    start_timecode = args.start_timecode
    max_sample = args.max_sample

    video_collection = {"PB": "Perfect Blue.mkv",
                        "MA": "Millennium Actress.rmvb"}

    video_name = video_collection[collection_name]

    video_file_path = project(video_name)

    # map(project, list(video_name.keys()))
    sampler = VideoSampler(video_file_path, sample_frequent=100)

    if args.mode == "single" or args.start_timecode != "00:00:00:00":
        sampler.oneSampling(start_timecode=start_timecode, sample_frequent=frequent, max_sample=max_sample)
    elif args.mode == "random":
        sampler.randomSampling(least_frequent=frequent, times=times, max_sample=max_sample)

    print('sampling all done.')


if __name__ == "__main__":
    main()
