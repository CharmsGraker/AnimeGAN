import tensorflow as tf
import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################



def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    """
        普通卷积层模块，设置了pad方案
    """
    with tf.variable_scope(scope):
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    """
    反卷积(转置卷积)
    :param x:
    :param channels:
    :param kernel:
    :param stride:
    :param use_bias:
    :param sn:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], tf.shape(x)[1] * stride, tf.shape(x)[2] * stride, channels]
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding='SAME')

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias)
            # 使用SAME填充

        return x


##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    """
    残差块(Residual Block)定义
    """
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


##################################################################################
# Sampling
##################################################################################

def flatten(x):
    return tf.layers.flatten(x)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x):
    return tf.sigmoid(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    """
        instance normal 对于每一个样本，沿深度方向进行normalization
        https://zhuanlan.zhihu.com/p/32610458
    """
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm'):
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def spectral_norm(w, iteration=1):
    """
        spectral_norm: 谱归一
        为了让判别器更优雅的满足利普西茨条件
        https://zhuanlan.zhihu.com/p/50567059
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)


    """
    # 奇异值分解
    # sigma(W)max = U * W.T * V
    # 也相当于对输入做了三次变换。 如果将输入视为一个向量的话，
    # 之后 [公式] 对应的拉伸变换才会改变输入的长度；其他两次旋转变换都不会。
    # 很显然，只需对sigma矩阵进行归一就好了。除以他的最大奇异值
    """
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


##################################################################################
# Loss function
##################################################################################
"""
没啥好说的,L1,L2正则化，Huber损失
"""
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def L2_loss(x, y):
    size = tf.size(x)
    return tf.nn.l2_loss(x - y) * 2 / tf.to_float(size)


def Huber_loss(x, y):
    return tf.losses.huber_loss(x, y)


def discriminator_loss(loss_func, real, gray, fake, real_blur):
    """
        得到三张图像。准备算loss
        https://blog.csdn.net/qq_32439305/article/details/87025405
    """

    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0
    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = -tf.reduce_mean(real)
        gray_loss = tf.reduce_mean(gray)
        fake_loss = tf.reduce_mean(fake)
        real_blur_loss = tf.reduce_mean(real_blur)
    """
    lsGANs损失。最小二乘GANs损失
        https://blog.csdn.net/victoriaw/article/details/60755698
    """
    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(real - 1.0))
        gray_loss = tf.reduce_mean(tf.square(gray))
        fake_loss = tf.reduce_mean(tf.square(fake))
        real_blur_loss = tf.reduce_mean(tf.square(real_blur))

    if loss_func == 'gan' or loss_func == 'dragan':
        """
            0/1标签
        """
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        gray_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(gray), logits=gray))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge':
        """
            对于Hinge，positive 应该接近1，negative应该接近-1
        """
        real_loss = tf.reduce_mean(relu(1.0 - real))
        gray_loss = tf.reduce_mean(relu(1.0 + gray))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    loss = 1.7 * real_loss + 1.7 * fake_loss + 1.7 * gray_loss + 0.8 * real_blur_loss

    return loss


def generator_loss(loss_func, fake):
    '''
    目的是让自己生成的图像标签越来越像真实图像的标签。即为真（1）

    :param loss_func:取决于GAN架构
    :param fake: generator生成的伪风格图像
    :return: loss
    '''
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss


def gram(x):
    """
    按道理x的话，本来有四维。
    但是这里不再区分图像的长宽，也就是把他flatten了
    查阅资料发现，这个叫gram matrix
    https://www.zhihu.com/question/49805962?from=profile_question_card

    也就是说得到的是个矩阵，这个矩阵可以量化两张图片的风格差异
    说白了，就是拉直了，然后计算他们的（偏）协方差矩阵
    :param x: 图像
    :return: 重组后的x
    """
    shape_x = tf.shape(x)
    # b 是样本数
    b = shape_x[0]

    # c是通道数
    c = shape_x[3]
    """
        重新组织X
        X shape: b * n * c
    """
    x = tf.reshape(x, [b, -1, c])
    """
        tf.cast()数据转型。
    """
    """
        X.T shape: b * c * n 
        X shape: b * n * c
        张量相乘
        https://blog.csdn.net/Dontla/article/details/103424954
    """
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


def con_loss(vgg, real, fake):
    """
    一种损失。用Vgg提取出来的特征图作为L1正则项
    :param vgg:
    :param real:
    :param fake:
    :return:
    """
    # 每次都重新run Vgg，以看当前图的feature map
    vgg.build(real)
    real_feature_map = vgg.conv4_4_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_4_no_activation

    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss


def style_loss(style, fake):
    """
    本loss用于计算风格损失
    :param style: 训练资料里的anime img
    :param fake: generator生成的img
    :return:
    """
    return L1_loss(gram(style), gram(fake))


def con_sty_loss(vgg, real, anime, fake):
    """
    还是利用Vgg模型来做为一个提取特征的模型，这里关注风格损失，和真伪损失
    :param vgg: vgg19
    :param real:真实图像。即图片重构的
    :param anime:红辣椒图像
    :param fake:generator生成的图像
    :return: c_loss:重构损失
    :return: s_loss:style(风格)损失，

    """
    vgg.build(real)
    real_feature_map = vgg.conv4_4_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_4_no_activation

    vgg.build(anime[:fake_feature_map.shape[0]])
    anime_feature_map = vgg.conv4_4_no_activation

    c_loss = L1_loss(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss


def color_loss(con, fake):
    """
    转换为YUV色彩编码
    color loss: 计算RGB三个通道的L1-正则项sum作为color loss
    """
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return L1_loss(con[:, :, :, 0], fake[:, :, :, 0]) + Huber_loss(con[:, :, :, 1], fake[:, :, :, 1]) + Huber_loss(
        con[:, :, :, 2], fake[:, :, :, 2])


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    至于为什么要转换成YUV，应该是因为RGB张成的三维空间有限吧。YUV则是整个立体空间，对于色彩都有意义
    什么是YUV: https://baike.baidu.com/item/YUV/3430784?fr=aladdin
    """
    rgb = (rgb + 1.0) / 2.0
    # rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
    #                                 [0.587, -0.331, -0.418],
    #                                 [0.114, 0.499, -0.0813]]]])
    # rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    # temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    # temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    # return temp
    return tf.image.rgb_to_yuv(rgb)
