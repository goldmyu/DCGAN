import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

tf.reset_default_graph()

# ----------------------Defining the models hyper-parameters--------------------------------------

learning_rate = 0.0002
momentum_beta1 = 0.5
batch_size = 209
epochs = 2000

# =================================== Configurations ===================================================================

model_save_flag = False
model_restore_flag = False
show_images = False

output_path_dir = "generated_files/flowers/"
if not os.path.exists(output_path_dir):
    os.makedirs(output_path_dir)

ckpt_path = output_path_dir + "checkpoints/model.ckpt"


# ------------------------------------ Models Definition ----------------------------------------

def generator(z, _training=True):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        # First layer - reshape to  4x4x1024  batch-normalized and relu activated
        dense_layer1 = tf.layers.dense(z, 1024 * 4 * 4)
        gen_layer1 = tf.reshape(dense_layer1, [-1, 4, 4, 1024])
        batch_norm1 = tf.layers.batch_normalization(gen_layer1, training=_training)
        activation_layer1 = tf.nn.relu(batch_norm1)

        # second layer - a de-conv to 8x8x512 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv2 = tf.layers.conv2d_transpose(activation_layer1, 512, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm2 = tf.layers.batch_normalization(gen_conv2, training=_training)
        activation_layer2 = tf.nn.relu(batch_norm2)

        # third layer - a de-conv to 16x16x256 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv3 = tf.layers.conv2d_transpose(activation_layer2, 256, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm3 = tf.layers.batch_normalization(gen_conv3, training=_training)
        activation_layer3 = tf.nn.relu(batch_norm3)

        # forth layer - a de-conv to 32x32x128 with stride of 2 and same padding, batch-normalized and relu activated
        gen_conv4 = tf.layers.conv2d_transpose(activation_layer3, 128, [5, 5], strides=(2, 2), padding='SAME')
        batch_norm4 = tf.layers.batch_normalization(gen_conv4, training=_training)
        activation_layer4 = tf.nn.relu(batch_norm4)

        # fifth layer- output - a de-conv to 64x64x1 with stride of 2 and same padding and tanh activated
        gen_conv5 = tf.layers.conv2d_transpose(activation_layer4, 3, [5, 5], strides=(2, 2), padding='SAME')
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


def plot_and_save_images(dims, img_label, generated_images, path, show=show_images):
    figure, subplots = plt.subplots(dims, dims, figsize=(dims, dims))
    figure.text(0.5, 0.05, img_label, ha='center')
    generated_images = 0.5 * generated_images + 0.5

    for iterator in range(dims * dims):
        i = iterator // dims
        j = iterator % dims
        subplots[i, j].get_xaxis().set_visible(False)
        subplots[i, j].get_yaxis().set_visible(False)
        subplots[i, j].cla()
        subplots[i, j].imshow(np.reshape(generated_images[iterator], (64, 64, 3)))
    if show:
        plt.show()
    plt.savefig(path)
    plt.close()


def save_model_to_checkpoint(flag=model_save_flag):
    if flag:
        try:
            save_path = saver.save(sess, ckpt_path)
            print("Model saved in path: %s" % save_path)
        except Exception as e:
            print("\nERROR : Could not save the model due to -  " + str(e))


def restore_model_from_ckpt(restore=model_restore_flag):
    if restore:
        try:
            saver.restore(sess, ckpt_path)
            print("\nModel restored from latest checkpoint")
        except:
            print("could not restore model, starting from scratch...")


# -------------------------------------- Model Train and Test -----------------------------------------------


def load_images(path):
    # return array of images
    images_list = listdir(path)
    loaded_images = []
    for image in images_list:
        try:
            image1 = tf.keras.preprocessing.image.load_img(path + image)
            x = tf.keras.preprocessing.image.img_to_array(image1)
            loaded_images.append(np.asarray(x))
        except OSError:
            print("error uploading image")

    return loaded_images


def model_training():
    # Training of the model

    train_time = time.time()

    df = pd.DataFrame(columns=['epoch_num', 'g_loss', 'd_loss', 'd_loss_fake', 'd_loss_real', 'epoch_runtime'])

    print('\nStarting training of the DCGAN model...')
    path = '../data-sets/flower_images/'

    # your images in an array
    imgs = load_images(path)
    imgs = np.asarray(imgs)
    imgs = tf.image.resize_images(imgs, [64, 64]).eval()  # Resize images from 28x28 to 64x64

    # processed_images = (imgs - 0.5) / 0.5  # normalize the data to the range of tanH [-1,1]
    processed_images = imgs / 255.0  # normalize the data to the range of tanH [-1,1]
    for epoch in range(epochs):
        epoch_start_time = time.time()
        discriminator_losses = []
        discriminator_loss_real = []
        discriminator_loss_fake = []
        generator_losses = []

        np.random.shuffle(processed_images)  # shuffle the dataset to get random samples

        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))  # Create random noise z for Generator

        d_loss1, g_loss1, disc_optimizer1, gen_optimizer1d, d_loss_real_data1, d_loss_generated_data1 = sess.run(
                [d_loss, g_loss, disc_optimizer, gen_optimizer, d_loss_real_data, d_loss_generated_data],
                {x: processed_images, z: z_, training: True})

        if epoch % 100 == 0:
            print('Training stats: epoch %d/%d\n'
                      'Discriminator loss: %.3f\nGenerator loss: %.3f' %
                      (epoch + 1, epochs,d_loss1, g_loss1))
            save_train_results(epoch)

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

        save_model_to_checkpoint()

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

# ----------------TF Session and CheckPoint---------------------------------------------------------------

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
