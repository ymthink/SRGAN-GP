#encoding=utf-8

import sys
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append('./')

from Model import *

learning_rate = 1e-3
batch_size = 32
LAMBDA = 10
SIGMA = 1e-3

step_num = 10000

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("../MNIST_data", one_hot=True)

def train():
    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./backup/latest/'):
        os.mkdir('./backup/latest/')
    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    gan = SRGAN(28, 28, 1, LAMBDA, SIGMA, batch_size)
    global_step = tf.Variable(0, name='global_step')
    global_step_op = tf.assign(global_step, tf.add(global_step, 1))

    D_opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gan.D_loss, var_list=gan.D_params)
    G_opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gan.G_loss, var_list=gan.G_params)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    if tf.train.get_checkpoint_state('./backup/latest/'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest/')
        print('********Restore the latest trained parameters.********')

    for step in range(step_num):
        for _ in range(5):
            xs, _ = data.train.next_batch(batch_size)
            xs = np.reshape(xs, [-1, 28, 28, 1])
            _, l_d = sess.run([D_opt, gan.D_loss], feed_dict={gan.x:xs})

        xs, _ = data.train.next_batch(batch_size)
        xs = np.reshape(xs, [-1, 28, 28, 1])
        _, l_g = sess.run([G_opt, gan.G_loss], feed_dict={gan.x:xs})
        sess.run(global_step_op)
        s = sess.run(global_step)

        if step % 100 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './backup/latest/', write_meta_graph=False)
            xs = np.reshape(data.test.images[random.randint(0, 9999)], [-1, 28, 28, 1])
            zs, gs = sess.run([gan.z, gan.g], feed_dict={gan.x:xs})
            fig = show_result(xs[0], zs[0], gs[0])
            #fig.set_size_inches(18.5, 10.5)
            plt.savefig('out/{}.png'.format(str(s).zfill(10)), bbox_inches='tight')
            plt.close(fig)
            print('********step: {}, D_loss = {:.8f}, G_loss = {:.8f}********'.format(s, l_d, l_g))
            
def show_result(xs, zs, gs):
    zs = np.reshape(zs, [7, 7])
    xs = np.reshape(xs, [28, 28])
    gs = np.reshape(gs, [28, 28])
    fig = plt.figure(figsize=(5, 15))

    #graph = gridspec.GridSpec(1, 3)
    #graph.update(wspace=0.5, hspace=0.5)

    ax = fig.add_subplot(131)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(zs, cmap='Greys_r')

    ax = fig.add_subplot(132)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(gs, cmap='Greys_r')

    ax = fig.add_subplot(133)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(xs, cmap='Greys_r')

    return fig



if __name__ == '__main__':
    train()



    

