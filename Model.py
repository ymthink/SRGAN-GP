'''An example for image SR using GAN
<Photo-Realistic Single Image Super-Resolution 
Using a Generative Adversarial Network>

https://arxiv.org/abs/1609.04802

E-mail: yinmiaothink@gmail.com
'''
import numpy as np
import tensorflow as tf

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def relu(x):
    return tf.nn.relu(x)

def elu(x):
    return tf.nn.elu(x)

def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)

def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size**2
    fan_out = output_dim * filter_size**2 / (stride**2)
    stddev = tf.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)

class Network(object):
    def __init__(self):
        self.layer_num = 0
        self.weights = []
        self.biases = []

    def conv2d(self, input, input_dim, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('conv'+str(self.layer_num)):

            init_w = he_init([filter_size, filter_size, input_dim, output_dim], stride)
            weight = tf.get_variable(
                'weight', 
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d(
                input,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output
    def deconv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('deconv'+str(self.layer_num)):
            input_shape = input.get_shape().as_list()
            init_w = he_init([filter_size, filter_size, output_dim, input_shape[3]], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d_transpose(
                value=input,
                filter=weight,
                output_shape=[
                    tf.shape(input)[0], 
                    input_shape[1]*stride, 
                    input_shape[2]*stride, 
                    output_dim
                ],
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)
            output = tf.reshape(output, [tf.shape(input)[0], input_shape[1]*stride, input_shape[2]*stride, output_dim])

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def batch_norm(self, input, scale=False):
        ''' batch normalization
        ArXiv 1502.03167v3 '''
        with tf.variable_scope('batch_norm'+str(self.layer_num)):
            output = tf.contrib.layers.batch_norm(input, scale=scale)
            self.layer_num += 1

        return output

    def dense(self, input, output_dim):
        with tf.variable_scope('dense'+str(self.layer_num)):
            input_dim = input.get_shape().as_list()[1]

            init_w = xavier_init([input_dim, output_dim])
            weight = tf.get_variable('weight', initializer=init_w)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b)

            output = tf.add(tf.matmul(input, weight), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def residual_block(self, input, output_dim, filter_size, n_blocks=5):
        output = input
        with tf.variable_scope('residual_block'):
            for i in range(n_blocks):
                bypass = output
                output = self.deconv2d(output, output_dim, filter_size, 1)
                output = self.batch_norm(output)
                output = tf.nn.relu(output)

                output = self.deconv2d(output, output_dim, filter_size, 1)
                output = self.batch_norm(output)
                output = tf.add(output, bypass)

        return output

    def pixel_shuffle(self, x, r, n_split):
        def PS(x, r):
            bs, a, b, c = x.get_shape().as_list()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, (bs, a, b, r, r))
            x = tf.transpose(x, (0, 1, 2, 4, 3))
            x = tf.split(x, a, 1)
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            x = tf.split(x, b, 1)
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            return tf.reshape(x, (bs, a*r, b*r, 1))

        xc = tf.split(x, n_split, 3)
        xc = tf.concat([PS(x_, r) for x_ in xc], 3)
        return xc

class SRGAN(object):
    def __init__(self, height, width, channel_num, LAMBDA, SIGMA, batch_size):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        #self.learning_rate = learning_rate
        self.channel_num = channel_num
        #self.vgg = VGG19(None, None, None)
        self.LAMBDA = LAMBDA
        self.SIGMA = SIGMA

        self.creat_model()

    def generator(self, z):
        G = Network()
        #Network.deconv2d(input, input_shape, output_dim, filter_size, stride)
        h = tf.nn.relu(G.deconv2d(z, 64, 3, 1))
        bypass = h


        h = G.residual_block(h, 64, 3, 2)

        h = G.deconv2d(h, 64, 3, 1)
        h = G.batch_norm(h)
        h = tf.add(h, bypass)

        h = G.deconv2d(h, 256, 3, 1)
        h = G.pixel_shuffle(h, 2, 64)
        h = tf.nn.relu(h)

        h = G.deconv2d(h, 64, 3, 1)
        h = G.pixel_shuffle(h, 2, 16)
        h = tf.nn.relu(h)

        h = G.deconv2d(h, self.channel_num, 3, 1)
        
        self.G_params = G.weights+G.biases

        return h

    def discriminator(self, x):
        D = Network()
        #Network.conv2d(input, output_dim, filter_size, stride, padding='SAME')
        h = D.conv2d(x, self.channel_num, 64, 3, 1)
        h = lrelu(h)

        #h = D.conv2d(h, 64, 64, 3, 1)
        #h = lrelu(h)
        #h = D.batch_norm(h)

        map_nums = [64, 128, 256]

        for i in range(len(map_nums)-1):
            h = D.conv2d(h, map_nums[i], map_nums[i+1], 3, 1)
            h = lrelu(h)
            h = D.batch_norm(h)

            h = D.conv2d(h, map_nums[i+1], map_nums[i+1], 3, 2)
            h = lrelu(h)
            h = D.batch_norm(h)

        h_shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, h_shape[1]*h_shape[2]*h_shape[3]])
        h = D.dense(h, 1024)
        h = lrelu(h)

        h = D.dense(h, 1)

        self.D_params = D.weights+D.biases
        
        return h
    def downscale(self, x, K):
        mat = np.zeros([K, K, self.channel_num, self.channel_num])
        for i in range(self.channel_num):
            mat[:, :, i, i] = 1.0 / K**2
        filter = tf.constant(mat, dtype=tf.float32)
        return tf.nn.conv2d(x, filter, strides=[1, K, K, 1], padding='SAME')

    def vgg19_loss(self, x , g):
        _, real_phi = self.vgg.build_model(x, tf.constant(False), False)
        _, fake_phi = self.vgg.build_model(g, tf.constant(False), True)

        loss = None
        for i in range(len(real_phi)):
            l2_loss = tf.nn.l2_loss(real_phi[i] - fake_phi[i])
            if loss is None:
                loss = l2_loss
            else:
                loss += l2_loss

        return tf.reduce_mean(loss)

    def reconstruction_loss(self, x, g):
        return tf.reduce_sum(tf.square(x - g))

    def creat_model(self):
        self.x = tf.placeholder(
            tf.float32,
            [None, self.height, self.width, self.channel_num],
            name='x'
        )
        self.z = self.downscale(self.x, 4)

        with tf.variable_scope('generator'):
            self.g = self.generator(self.z)
        with tf.variable_scope('discriminator') as scope:
            self.D_real = self.discriminator(self.x)
            scope.reuse_variables()
            self.D_fake = self.discriminator(self.g)

        content_loss = self.reconstruction_loss(self.x, self.g)

        disc_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        gen_loss = -tf.reduce_mean(self.D_fake)

        alpha = tf.random_uniform(
            shape=[self.batch_size,1],
            minval=0.,
            maxval=1.
        )

        x_ = tf.reshape(self.x, [-1, self.height*self.width])
        g_ = tf.reshape(self.g, [-1, self.height*self.width])

        differences = x_ - g_
        interpolates = x_ + alpha * differences
        interpolates = tf.reshape(interpolates, [-1, self.height, self.width, self.channel_num])
        gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.D_loss = self.SIGMA * (disc_loss + self.LAMBDA * gradient_penalty)

        self.G_loss = content_loss + self.SIGMA * gen_loss



        


        





        




