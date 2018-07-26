import itertools
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


def generator(z, training=True):
    with tf.variable_scope('Generator', reuse=False):

        # First layer - reshape to  4x4x1024  batch-normalized and relu activated
        dense_layer1 = tf.layers.dense(z, 1024 * 4 * 4)
        gen_layer1 = tf.reshape(dense_layer1, [-1, 4, 4, 1024])
        batch_norm1 = tf.layers.batch_normalization(gen_layer1, training=training)
        activation_layer1 = tf.nn.relu(batch_norm1)

        # second layer - a de-conv to 8x8x512 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv2 = tf.layers.conv2d_transpose(activation_layer1, 512, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm2 = tf.layers.batch_normalization(gen_conv2, training=training)
        activation_layer2 = tf.nn.relu(batch_norm2)

        # third layer - a de-conv to 16x16x256 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv3 = tf.layers.conv2d_transpose(activation_layer2, 256, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm3 = tf.layers.batch_normalization(gen_conv3, training=training)
        activation_layer3 = tf.nn.relu(batch_norm3)

        # forth layer - a de-conv to 32x32x128 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv4 = tf.layers.conv2d_transpose(activation_layer3, 128, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm4 = tf.layers.batch_normalization(gen_conv4, training=training)
        activation_layer4 = tf.nn.relu(batch_norm4)

        # fifth layer- output - a de-conv to 64x64x3 with stride of 2 and same padding and tanh activated
        gen_conv5 = tf.layers.conv2d_transpose(activation_layer4, 1, [5, 5], strides=(2, 2), padding='SAME')
        activation_layer5 = tf.tanh(gen_conv5)
        return activation_layer5


def discriminator(x, reuse, training=True):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # First layer - conv to  32x32x128  with stride of 2 and same padding  leaky-relu activated
        disc_conv1 = tf.layers.conv2d(x, 128, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv1, training=training)
        disc_activation1 = leaky_relu(batch_norm_disc)
        #         disc_activation1 = leaky_relu(disc_conv1)

        # Second layer - conv to  16x16x256  with stride of 2 and same padding, batch-normalized leaky-relu activated
        disc_conv2 = tf.layers.conv2d(disc_activation1, 256, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv2, training=training)
        disc_activation2 = leaky_relu(batch_norm_disc)

        # Third layer - conv to  8x8x512 with stride of 2 and same padding, batch-normalized leaky-relu activated
        disc_conv3 = tf.layers.conv2d(disc_activation2, 512, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv3, training=training)
        disc_activation3 = leaky_relu(batch_norm_disc)

        # Forth layer - conv to  4x4x1024 with stride of 2 and same padding, batch-normalized leaky-relu activated
        disc_conv4 = tf.layers.conv2d(disc_activation3, 1024, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv4, training=training)
        disc_activation4 = leaky_relu(batch_norm_disc)

        # Output layer - conv to  1  sigmoid activated
        disc_conv5 = tf.layers.conv2d(disc_activation4, 1, [4, 4])
        # disc_output = tf.nn.sigmoid(disc_conv5)
        return disc_conv5


def leaky_relu(x):
    return tf.maximum(0.2 * x, x)


def show_result(num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(generated, feed_dict={z: np.random.normal(0, 1, (25, 1, 1, 100)), training: False})

    size_figure_grid = 5

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------------

learning_rate = 0.0002
momentum_beta1 = 0.5
batch_size = 128
epochs = 5
#weight_init_std = 0.02

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 100], name='Z')
x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='X')
training = tf.placeholder(dtype=tf.bool)

generated = generator(z, training)

disc_logits_real = discriminator(x, reuse=False)
disc_logits_fake = discriminator(generated, reuse=True)

d_labels_real = tf.ones_like(disc_logits_real)
d_labels_fake = tf.zeros_like(disc_logits_fake)

d_loss_real_data = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_real, logits=disc_logits_real)
d_loss_generated_data = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_fake, logits=disc_logits_fake)

d_loss = tf.reduce_mean(d_loss_real_data + d_loss_generated_data)

# loss of generator is to get discriminator say for every generated image that it is 1 - not fake
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_fake), logits=disc_logits_fake))

all_vars = tf.trainable_variables()
disc_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
generator_vars = [var for var in all_vars if var.name.startswith('Generator')]

# optimizer for each network
disc_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=momentum_beta1).minimize(d_loss, var_list=disc_vars)
gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=momentum_beta1).minimize(g_loss, var_list=generator_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()
try:
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
except:
    print("could not restore model, starting from scratch...")

print('\nStarting training of the DCGAN model...')

num_of_iterations = mnist.train.num_examples // batch_size
for epoch in range(epochs):
    discriminator_losses = []
    generator_losses = []
    for i in range(num_of_iterations):
        if i % 100 == 0:
            print('Training stats: iteration number %d/%d in epoch number %d' % (i, num_of_iterations, epoch + 1))
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)

        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        x_batch = mnist.train.next_batch(batch_size)
        x_ = tf.image.resize_images(x_batch[0], [64, 64]).eval()

        d_loss1, g_loss1, disc_optimizer1, gen_optimizer1 = sess.run([d_loss, g_loss, disc_optimizer, gen_optimizer],
                                                                     {x: x_, z: z_, training: True})

        discriminator_losses.append(d_loss1)
        generator_losses.append(g_loss1)

    print('Training epoch number %d out of %d - loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), epochs, np.mean(discriminator_losses), np.mean(generator_losses)))

    train_results_dir = "train_results/"
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)

    show_result(epoch + 1, show=False, save=True, path='train_results/' + str(epoch + 1) + '.png')
    print("path of images: train_results/" + str(epoch + 1) + '.png')

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

sess.close()
