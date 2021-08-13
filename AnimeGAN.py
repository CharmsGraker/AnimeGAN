from tools.ops import *
from tools.utils import *
from glob import glob
import time
import numpy as np
from net import generator
from net.discriminator import D_net
from tools.data_loader import ImageGenerator
from tools.vgg19 import Vgg19


class AnimeGAN(object):
    def __init__(self, sess, args):
        self.model_name = 'AnimeGAN'

        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.data_mean = args.data_mean

        self.epoch = args.epoch
        self.init_epoch = args.init_epoch  # args.epoch // 20

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size

        self.save_freq = args.save_freq

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        """ Weight """
        # 混合权重
        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.sty_weight = args.sty_weight
        self.color_weight = args.color_weight

        self.training_rate = args.training_rate
        self.ld = args.ld

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch

        # 是否使用谱归一(spectral norm)
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # Sample * Width * Height * Channel
        # https://blog.csdn.net/kdongyi/article/details/82343712
        self.real = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch],
                                   name='real_A')
        self.anime = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch],
                                    name='anime_A')
        self.anime_smooth = tf.placeholder(tf.float32,
                                           [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch],
                                           name='anime_smooth_A')
        self.test_real = tf.placeholder(tf.float32, [1, None, None, self.img_ch], name='test_input')

        self.anime_gray = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch],
                                         name='anime_B')

        self.real_image_generator = ImageGenerator('./dataset/train_photo', self.img_size, self.batch_size,
                                                   self.data_mean)
        self.anime_image_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/style'), self.img_size,
                                                    self.batch_size, self.data_mean)
        self.anime_smooth_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/smooth'),
                                                     self.img_size, self.batch_size, self.data_mean)
        # 得到最多的训练集资料数目。取决于真实图像数或风格化动漫图像数目最大的那个
        self.dataset_num = max(self.real_image_generator.num_images, self.anime_image_generator.num_images)

        # 训练好的vgg19模型，用于提取图片潜在特征，然后方便计算各种损失
        self.vgg = Vgg19()

        print()
        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight : ", self.g_adv_weight, self.d_adv_weight,
              self.con_weight, self.sty_weight, self.color_weight)
        print("# init_lr,g_lr,d_lr : ", self.init_lr, self.g_lr, self.d_lr)
        print(f"# training_rate G -- D: {self.training_rate} : 1")
        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):

        with tf.variable_scope(scope, reuse=reuse):
            G = generator.G_net(x_init)
            return G.fake

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        """

        :param x_init:
        :param reuse:
        :param scope:
        :return: 返回一判别器网络
        """
        D = D_net(x_init, self.ch, self.n_dis, self.sn, reuse=reuse, scope=scope)

        return D

    ##################################################################################
    # Model
    ##################################################################################
    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        # 做一个插值图像
        interpolated = real + alpha * (fake - real)

        logit, _ = self.discriminator(interpolated, reuse=True, scope=scope)

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        GP = 0
        # WGAN - LP
        if self.gan_type.__contains__('lp'):
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type.__contains__('gp') or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def build_model(self):

        """ Define Generator, Discriminator """
        """
            定义损失
        """
        self.generated = self.generator(self.real)
        self.test_generated = self.generator(self.test_real, reuse=True)

        anime_logit = self.discriminator(self.anime)
        anime_gray_logit = self.discriminator(self.anime_gray, reuse=True)

        generated_logit = self.discriminator(self.generated, reuse=True)
        smooth_logit = self.discriminator(self.anime_smooth, reuse=True)

        """ Define Loss """
        """
            gradient panalty
        """
        if self.gan_type.__contains__('gp') or self.gan_type.__contains__('lp') or self.gan_type.__contains__('dragan'):
            GP = self.gradient_panalty(real=self.anime, fake=self.generated)
        else:
            GP = 0.0

        # init pharse
        init_c_loss = con_loss(self.vgg, self.real, self.generated)
        # 以重构损失作为初始损失
        init_loss = self.con_weight * init_c_loss

        self.init_loss = init_loss

        # gan
        c_loss, s_loss = con_sty_loss(self.vgg, self.real, self.anime_gray, self.generated)
        t_loss = self.con_weight * c_loss + self.sty_weight * s_loss + color_loss(self.real,
                                                                                  self.generated) * self.color_weight

        # 定义了超参数g_adv_weight，用于对generator loss 赋予权重
        g_loss = self.g_adv_weight * generator_loss(self.gan_type, generated_logit)
        d_loss = self.d_adv_weight * discriminator_loss(self.gan_type, anime_logit, anime_gray_logit, generated_logit,
                                                        smooth_logit) + GP
        """
            generator不仅得考虑图像重构损失，还得考虑风格损失。这样他就会去寻找一个balance的点
        """
        self.Generator_loss = t_loss + g_loss
        self.Discriminator_loss = d_loss

        """ Training """
        """
                得到所有变量
        """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        # 注意，这里标明了tf.train，表示这是用于训练的，那么会BP
        # 在之后会看到出现在sess.run()里
        # 初始化Adam优化器，以及他需要(最小化)的参数列表。通过优化器都是寻找最小
        self.init_optim = tf.train.AdamOptimizer(self.init_lr, beta1=0.5, beta2=0.999).minimize(self.init_loss,
                                                                                                var_list=G_vars)
        # 初始化generator的Adam优化器
        self.G_optim = tf.train.AdamOptimizer(self.g_lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss,
                                                                                          var_list=G_vars)
        # 初始化discriminator的Adam优化器
        self.D_optim = tf.train.AdamOptimizer(self.d_lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss,
                                                                                          var_list=D_vars)

        """" Summary """
        """
            用于展示各种summary指标
        """
        self.G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_gan = tf.summary.scalar("G_gan", g_loss)
        self.G_vgg = tf.summary.scalar("G_vgg", t_loss)
        self.G_init_loss = tf.summary.scalar("G_init", init_loss)

        self.V_loss_merge = tf.summary.merge([self.G_init_loss])
        self.G_loss_merge = tf.summary.merge([self.G_loss, self.G_gan, self.G_vgg, self.G_init_loss])
        self.D_loss_merge = tf.summary.merge([self.D_loss])

    def train(self):
        """
            train里只详细说明了该以怎么方式和频率训练。具体损失怎么反馈的，得看build_model
        :return: 无返回
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # saver to save model
        # 定义一个saver，为方便保存断点使用
        self.saver = tf.train.Saver(max_to_keep=self.epoch)

        # summary writer
        # 定义summary Writer存储路径，为之后存储各种summary指标
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        """ Input Image"""
        # 从这里读入图像数据
        # 共三组，每一组两张：A,B
        real_img_op, anime_img_op, anime_smooth_op = self.real_image_generator.load_images(), self.anime_image_generator.load_images(), self.anime_smooth_generator.load_images()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1

            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0

            print(" [!] Load failed...")

        # loop for epoch
        init_mean_loss = []
        mean_loss = []
        # training times , G : D = self.training_rate : 1
        j = self.training_rate
        # 训练从上次训练epoch数之后，到max epoch数
        for epoch in range(start_epoch, self.epoch):

            # idx代表了需要几次batch
            for idx in range(int(self.dataset_num / self.batch_size)):

                # https://blog.csdn.net/huahuazhu/article/details/76178681
                # 告诉tf，想要这些节点的输出，也就是这些节点可以flow了。
                # 这里flow的是数据集的迭代器，所以就是取出下一批训练资料的color和gray
                anime, anime_smooth, real = self.sess.run([anime_img_op, anime_smooth_op, real_img_op])

                # 定义train需要的参数与输入
                # key通过tf.placeholder在上文已经定义了
                train_feed_dict = {
                    self.real: real[0],
                    self.anime: anime[0],
                    self.anime_gray: anime[1],
                    self.anime_smooth: anime_smooth[1]
                }

                # 小于需要初始的epoch数，那么先init generator
                if epoch < self.init_epoch:
                    # Init G
                    start_time = time.time()
                    # 表示希望得到右侧列表里的那些变量当前流动状态
                    # 传入init_optim 为了告诉tf这是在训练，允许BP
                    real_images, generator_images, _, v_loss, summary_str = self.sess.run([self.real, self.generated,
                                                                                           self.init_optim,
                                                                                           self.init_loss,
                                                                                           self.V_loss_merge],
                                                                                          feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, epoch)
                    init_mean_loss.append(v_loss)

                    print("Epoch: %3d Step: %5d / %5d  time: %f s init_v_loss: %.8f  mean_v_loss: %.8f" % (
                        epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, v_loss,
                        np.mean(init_mean_loss)))
                    # 清理一波mean loss
                    if (idx + 1) % 200 == 0:
                        init_mean_loss.clear()
                # epoch >= self.init_epoch:
                else:
                    start_time = time.time()

                    # GAN不能疯狂训练判别器，判别器训练太频繁会均衡不好。
                    if j == self.training_rate:
                        # Update D
                        # 训练discriminator，那么传入D的优化器
                        _, d_loss, summary_str = self.sess.run(
                            [self.D_optim, self.Discriminator_loss, self.D_loss_merge],
                            feed_dict=train_feed_dict)
                        self.writer.add_summary(summary_str, epoch)

                    # Update G
                    # 训练时，记得传入G的优化器
                    real_images, generator_images, _, g_loss, summary_str = self.sess.run(
                        [self.real, self.generated, self.G_optim,
                         self.Generator_loss, self.G_loss_merge], feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, epoch)

                    mean_loss.append([d_loss, g_loss])

                    # 显示epoch训练情况
                    if j == self.training_rate:
                        print(
                            "Epoch: %3d Step: %5d / %5d  time: %f s d_loss: %.8f, g_loss: %.8f -- mean_d_loss: %.8f, mean_g_loss: %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, d_loss,
                                g_loss, np.mean(mean_loss, axis=0)[0],
                                np.mean(mean_loss, axis=0)[1]))
                    else:
                        print(
                            "Epoch: %3d Step: %5d / %5d time: %f s , g_loss: %.8f --  mean_g_loss: %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, g_loss,
                                np.mean(mean_loss, axis=0)[1]))

                    if (idx + 1) % 200 == 0:
                        mean_loss.clear()

                    j = j - 1
                    if j < 1:
                        j = self.training_rate

            # 保存断点
            if (epoch + 1) >= self.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            # 可以保持image了
            if epoch >= self.init_epoch - 1:
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                for i, sample_file in enumerate(val_files):
                    print('val: ' + str(i) + sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))

                    # 使用test集
                    # 这里没有传入优化器，因为是在做test
                    test_real, test_generated = self.sess.run([self.test_real, self.test_generated],
                                                              feed_dict={self.test_real: sample_image})

                    save_images(test_real, save_path + '{:03d}_a.png'.format(i), None)
                    # adjust_brightness_from_photo_to_fake
                    save_images(test_generated, save_path + '{:03d}_b.png'.format(i), sample_file)

    @property
    def model_dir(self):
        """
                模型名称：AnimeGAN
                使用的训练集名称
                GAN类型

        :return:
        """
        return "{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                self.gan_type,
                                                int(self.g_adv_weight), int(self.d_adv_weight), int(self.con_weight),
                                                int(self.sty_weight), int(self.color_weight))

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        """
        可能训练非常缓慢，或者说当前条件不允许一次性训练完。这个函数使得可以从上次check point(断点）处继续训练
        :param checkpoint_dir: 断点保存路径
        :return: flag:True/False,counter:last epoch
        """
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        # 初始化所有变量
        tf.global_variables_initializer().run()
        test_files = glob('./dataset/{}/*.*'.format('test/test_photo'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in test_files:
            print('Processing the image: ', sample_file)
            sample_image = np.asarray(load_test_data(sample_file, self.img_size))
            image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(sample_file)))

            # 因为没有传入优化器所以就知道这是test
            # 好像很有道理.....
            fake_img = self.sess.run(self.test_generated, feed_dict={self.test_real: sample_image})
            # adjust_brightness_from_photo_to_fake
            save_images(fake_img, image_path, sample_file)

            index.write("<td>%s</td>" % os.path.basename(image_path))

            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_size[1] + 32, self.img_size[0] + 32))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_size[1] + 32, self.img_size[0] + 32))
            index.write("</tr>")

        index.close()
