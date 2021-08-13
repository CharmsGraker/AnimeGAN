import cv2, argparse, os
from glob import glob
from tqdm import tqdm


def parse_args():
    desc = "get the mean values of  b,g,r on the whole dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')

    return parser.parse_args()


def read_img(image_path):
    img = cv2.imread(image_path)
    assert len(img.shape) == 3
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()
    return B, G, R


def get_mean(dataset_name):
    """
        请进入tools目录下运行
    :param dataset_name:
    :return:
    """
    # 如果是以进入tools目录运行 则os.getcwd() 为 D:\Gary\Program\GitHubProject\AnimeGAN\tools
    # print('os.pardir: ', os.pardir) # 受启动方式有很大影响
    assert os.path.basename(os.getcwd()) == 'tools'

    # 用于查找
    file_list = glob(os.pardir + '/dataset/{}/{}/*.*'.format(dataset_name, 'style'))
    image_num = len(file_list)
    print('image_num:', image_num)

    B_total = 0
    G_total = 0
    R_total = 0
    for f in tqdm(file_list):
        bgr = read_img(f)
        B_total += bgr[0]
        G_total += bgr[1]
        R_total += bgr[2]

    B_mean, G_mean, R_mean = B_total / image_num, G_total / image_num, R_total / image_num
    mean = (B_mean + G_mean + R_mean) / 3

    return mean - B_mean, mean - G_mean, mean - R_mean


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    return get_mean(args.dataset)


if __name__ == '__main__':
    result = main()
    print('style_data_mean_diff (B, G, R):  ', result)
