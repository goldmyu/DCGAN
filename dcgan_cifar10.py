import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10

tf.reset_default_graph()

# =============================== Defining the models hyper-parameters =================================================

learning_rate = 0.0002
momentum_beta1 = 0.5
epochs = 50
batch_size = 30

# =================================== Configurations ===================================================================

model_save_flag = False
model_restore_flag = False
show_images = False

output_path_dir = "generated_files/cifar10/"
ckpt_path = output_path_dir + "checkpoints/model.ckpt"

if not os.path.exists(output_path_dir):
    os.makedirs(output_path_dir)


# =====================================  Models Definition =============================================================


def generator(z, _training=True):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        # First layer - reshape to  4x4x1024  batch-normalized and relu activated
        dense_layer1 = tf.layers.dense(inputs=z, units=1024 * 4 * 4)
        gen_layer1 = tf.reshape(tensor=dense_layer1, shape=[-1, 4, 4, 1024])
        batch_norm1 = tf.layers.batch_normalization(inputs=gen_layer1, training=_training)
        activation_layer1 = tf.nn.relu(features=batch_norm1)

        # second layer - a de-conv to 8x8x512 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv2 = tf.layers.conv2d_transpose(inputs=activation_layer1, filters=512,
                                               kernel_size=[5, 5], strides=(2, 2), padding='SAME')

        batch_norm2 = tf.layers.batch_normalization(inputs=gen_conv2, training=_training)
        activation_layer2 = tf.nn.relu(batch_norm2)

        # third layer - a de-conv to 16x16x256 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv3 = tf.layers.conv2d_transpose(inputs=activation_layer2, filters=256,
                                               kernel_size=[5, 5], strides=(2, 2), padding='SAME')

        batch_norm3 = tf.layers.batch_normalization(inputs=gen_conv3, training=_training)
        activation_layer3 = tf.nn.relu(batch_norm3)

        # forth layer - a de-conv to 32x32x128 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv4 = tf.layers.conv2d_transpose(inputs=activation_layer3, filters=128, kernel_size=[5, 5],
                                               strides=(2, 2), padding='SAME')

        batch_norm4 = tf.layers.batch_normalization(inputs=gen_conv4, training=_training)
        activation_layer4 = tf.nn.relu(batch_norm4)

        # fifth layer- output - a de-conv to 64x64x3 with stride of 2 and same padding and tanh activated
        gen_conv5 = tf.layers.conv2d_transpose(inputs=activation_layer4, filters=3, kernel_size=[5, 5],
                                               strides=(2, 2), padding='SAME')
        activation_layer5 = tf.tanh(gen_conv5)
        return activation_layer5


def discriminator(x, _training=True):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # First layer - conv to 32x32x128, stride of 2, same padding, batch-normalization and leaky-relu activated
        disc_conv1 = tf.layers.conv2d(x, 128, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv1, training=_training)
        disc_activation1 = tf.nn.leaky_relu(batch_norm_disc)

        # Second layer - conv to  16x16x256  with stride of 2 and same padding, batch-normalized leaky-relu activated
        disc_conv2 = tf.layers.conv2d(disc_activation1, 256, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv2, training=_training)
        disc_activation2 = tf.nn.leaky_relu(batch_norm_disc)

        # Third layer - conv to  8x8x512 with stride of 2 and same padding, batch-normalized leaky-relu activated
        disc_conv3 = tf.layers.conv2d(inputs=disc_activation2, filters=512, kernel_size=[5, 5], strides=(2, 2),
                                      padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv3, training=_training)
        disc_activation3 = tf.nn.leaky_relu(batch_norm_disc)

        # Forth layer - conv to  4x4x1024 with stride of 2 and same padding, batch-normalized leaky-relu activated
        disc_conv4 = tf.layers.conv2d(disc_activation3, 1024, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm_disc = tf.layers.batch_normalization(disc_conv4, training=_training)
        disc_activation4 = tf.nn.leaky_relu(batch_norm_disc)

        # Output layer - conv to 1x1x1
        disc_conv5 = tf.layers.conv2d(disc_activation4, 1, [4, 4])
        return disc_conv5


# ---------------------------------------------------------------------------------------------


def save_train_results(epoch_num):
    path = output_path_dir + '/epoch' + str(epoch_num + 1) + '.png'
    dims = 4
    z_ = np.random.normal(0, 1, (16, 1, 1, 100))
    generated_images = sess.run(generated, feed_dict={z: z_, training: False})
    img_label = 'Generated images after {} training epoch'.format(epoch_num + 1)
    plot_and_save_images(dims, img_label, generated_images, path)


def plot_and_save_images(dims, img_label, generated_images, path, show=show_images, save=True):
    figure, subplots = plt.subplots(nrows=dims, ncols=dims, figsize=(dims, dims))
    figure.text(0.5, 0.05, img_label, ha='center')
    generated_images = 0.5 * generated_images + 0.5
    for iterator in range(dims * dims):
        i = iterator // dims
        j = iterator % dims
        subplots[i, j].get_xaxis().set_visible(False)
        subplots[i, j].get_yaxis().set_visible(False)
        subplots[i, j].cla()
        subplots[i, j].imshow(np.reshape(a=generated_images[iterator], newshape=(64, 64, 3)))
    if show:
        plt.show()
    if save:
        plt.savefig(path)
    plt.close()


def save_model_to_checkpoint(model_save=model_save_flag):
    if model_save:
        try:
            save_path = saver.save(sess, ckpt_path)
            print("Model saved in path: %s" % save_path)
        except Exception as e:
            print("\nERROR : Could not save the model due to -  " + str(e))


def restore_model_from_ckpt(model_restore=model_restore_flag):
    if model_restore:
        try:
            saver.restore(sess, ckpt_path)
            print("\nModel restored from latest checkpoint")
        except:
            print("could not restore model, starting from scratch...")


# -------------------------------------- Model Train and Test -----------------------------------------------


def model_training():
    # Training of the model

    train_time = time.time()
    df = pd.DataFrame(columns=['epoch_num', 'g_loss', 'd_loss', 'd_loss_fake', 'd_loss_real', 'epoch_runtime'])

    num_of_iterations = len(x_train) // batch_size

    print('\nStarting training of the DCGAN model...')
    for epoch in range(epochs):
        epoch_start_time = time.time()
        discriminator_losses = []
        discriminator_loss_real = []
        discriminator_loss_fake = []
        generator_losses = []

        random_shuffle_data = np.random.permutation(len(x_train))

        for _iter in range(num_of_iterations):
            indices = random_shuffle_data[_iter * batch_size: (_iter + 1) * batch_size]

            # Prepare train images - Resize images from 32x32 to 64x64
            train_data_batch = np.take(x_train, indices=indices, axis=0)
            train_data_batch = tf.image.resize_images(train_data_batch, [64, 64]).eval()

            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))  # Create random noise z for Generator

            d_loss1, g_loss1, disc_optimizer1, gen_optimizer1, d_loss_real_data1, d_loss_generated_data1 = sess.run(
                [d_loss, g_loss, disc_optimizer, gen_optimizer, d_loss_real_data, d_loss_generated_data],
                {x: train_data_batch, z: z_, training: True})

            if _iter % 50 == 0:
                print('Training stats: iteration number %d/%d in epoch number %d\n'
                      'Discriminator loss: %.3f\nGenerator loss: %.3f' %
                      (_iter, num_of_iterations, epoch + 1, d_loss1, g_loss1))

            discriminator_losses.append(d_loss1)
            discriminator_loss_real.append(d_loss_real_data1)
            discriminator_loss_fake.append(d_loss_generated_data1)
            generator_losses.append(g_loss1)

        epoch_runtime = time.time() - epoch_start_time
        print('Training epoch %d/%d - Time for epoch: %d discriminator loss: %.3f, Generator loss: %.3f' % (
            (epoch + 1), epochs, epoch_runtime, np.mean(discriminator_losses), np.mean(generator_losses)))

        df = df.append(pd.Series([epoch + 1, np.mean(generator_losses), np.mean(discriminator_losses),
                                  np.mean(discriminator_loss_fake), np.mean(discriminator_loss_real), epoch_runtime],
                                 index=df.columns), ignore_index=True)

        save_train_results(epoch)

    print('Total Training time was: %d' % (time.time() - train_time))
    df.to_csv(output_path_dir + 'dataFrame.csv', index=False)


def model_test():
    z_test = np.random.normal(0, 1, (1000, 1, 1, 100))  # Create random noise z for Generator
    disc, gen = sess.run([disc_logits_fake, generated], feed_dict={z: z_test, training: False})

    good_imgs = np.size(np.where(tf.sigmoid(disc).eval() > 0.5)[0])

    print("Testing the model with 1000 generated images from the trained generator...\n"
          "Our trained discriminator classified %d out of 1000 as real images." % good_imgs)

    plot_and_save_images(8, "Generated images", gen, output_path_dir + "model_test_img.png", False)


# ----------------------------------------------------------------------------

# The MNIST data-set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255 - 0.5) / 0.5

# Create place holders for variable x,z,training
z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 100], name='Z')
x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='X')
training = tf.placeholder(dtype=tf.bool)

# Define the Generator model
generated = generator(z, _training=training)

# Define the Discriminator model
disc_logits_real = discriminator(x)
disc_logits_fake = discriminator(generated)

# Define labels for the discriminator training
d_labels_real = tf.ones_like(disc_logits_real)
d_labels_fake = tf.zeros_like(disc_logits_fake)

# Define loss for generator - generator goal is to get the discriminator to classify each generated image as real
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_fake), logits=disc_logits_fake))

# Define loss functions for the Discriminator
d_loss_real_data = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_real, logits=disc_logits_real)
d_loss_generated_data = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_fake, logits=disc_logits_fake)
d_loss = tf.reduce_mean(d_loss_real_data + d_loss_generated_data)

# Define the different variables for the Generator and Discriminator separately
all_vars = tf.trainable_variables()
disc_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
generator_vars = [var for var in all_vars if var.name.startswith('Generator')]

# Define optimizer for Generator and Discriminator
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    disc_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=momentum_beta1).minimize(d_loss, var_list=disc_vars)
    gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=momentum_beta1).minimize(g_loss,
                                                                                         var_list=generator_vars)

# ----------------TF Session and Saver---------------------------------------------------------------

# Create tf session and initialize all the variable
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Create a tf saver to enable training check-points and try to restore from previous ckpt if exist
saver = tf.train.Saver()
restore_model_from_ckpt()

# -------------------------------------------------------------------------------

# Train the model
model_training()
save_model_to_checkpoint(True)

# Test model performance
model_test()

# End the tf session
sess.close()
