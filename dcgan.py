import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()


def generator(z, training=True):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
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


def discriminator(x, reuse=False, training=True):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
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


def save_train_result_image(epoch_num, show=False, path='img.png'):
    dims = 4
    generated_images = sess.run(generated, feed_dict={z: np.random.normal(0, 1, (16, 1, 1, 100)), training: False})

    figure, subplots = plt.subplots(dims, dims, figsize=(dims, dims))

    for iterator in range(dims * dims):
        i = iterator // dims
        j = iterator % dims
        subplots[i, j].get_xaxis().set_visible(False)
        subplots[i, j].get_yaxis().set_visible(False)
        subplots[i, j].cla()
        subplots[i, j].imshow(np.reshape(generated_images[iterator], (64, 64)), cmap='gray')

    img_label = 'Generated image after {} training epoch'.format(epoch_num + 1)
    figure.text(0.5, 0.05, img_label, ha='center')

    if show:
        plt.show()

    plt.savefig(path)
    plt.close()


# ----------------------------------------------------------------------------

# Defining the models hyperparameters
learning_rate = 0.0002
momentum_beta1 = 0.5
batch_size = 128
epochs = 5

# The MNIST data-set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# Create place holders for variable x,z,training
z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 100], name='Z')
x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='X')
training = tf.placeholder(dtype=tf.bool)
reuse = tf.placeholder(dtype=tf.bool)

# Define the Generator model
generated = generator(z, training=training)

# Define the Generator model
disc_logits_real = discriminator(x)
disc_logits_fake = discriminator(generated, reuse=True)

# Define labels for the discriminator training
d_labels_real = tf.ones_like(disc_logits_real)
d_labels_fake = tf.zeros_like(disc_logits_fake)

# Define loss functions for the Discriminator
d_loss_real_data = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_real, logits=disc_logits_real)
d_loss_generated_data = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_fake, logits=disc_logits_fake)
d_loss = tf.reduce_mean(d_loss_real_data + d_loss_generated_data)

# Define loss for generator
# loss of generator is to get discriminator say for every generated image that it is "1" meaning not fake
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_fake), logits=disc_logits_fake))

# Define the different variables for the Generator and Discriminator separately
all_vars = tf.trainable_variables()
disc_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
generator_vars = [var for var in all_vars if var.name.startswith('Generator')]

# Define optimizer for Generator and Discriminator
disc_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=momentum_beta1).minimize(d_loss, var_list=disc_vars)
gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=momentum_beta1).minimize(g_loss, var_list=generator_vars)

# Create tf session and initalize all the variable
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Create a tf saver to enable training check-points and try to restore from previous ckpt if exist
saver = tf.train.Saver()
try:
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
except:
    print("could not restore model, starting from scratch...")

# Check if we are running on a GPU or CPU
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('\nGPU device found at: {}'.format(device_name))

# Training of the model
print('\nStarting training of the DCGAN model...')
num_of_iterations = mnist.train.num_examples // (batch_size * 5)

for epoch in range(epochs):
    discriminator_losses = []
    generator_losses = []
    for i in range(num_of_iterations):
        if i % 100 == 0:
            print('Training stats: iteration number %d/%d in epoch number %d' % (i, num_of_iterations, epoch + 1))
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)

        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))  # Create random noise z for Generator

        x_batch = mnist.train.next_batch(batch_size)
        x_ = tf.image.resize_images(x_batch[0], [64, 64]).eval()  # Resize images from 28x28 to 64x64
        x_ = (x_ - 0.5) / 0.5  # normalize the data to the range of tanH [-1,1]

        d_loss1, g_loss1, disc_optimizer1, gen_optimizer1 = sess.run([d_loss, g_loss, disc_optimizer, gen_optimizer],
                                                                     {x: x_, z: z_, training: True})

        discriminator_losses.append(d_loss1)
        generator_losses.append(g_loss1)

    print('Training epoch number %d out of %d - discriminator loss is : %.3f, Generator loss is: %.3f' % (
        (epoch + 1), epochs, np.mean(discriminator_losses), np.mean(generator_losses)))

    train_results_dir = "train_results/"
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)

    img_path = 'train_results/epoch' + str(epoch + 1) + '.png'
    save_train_result_image(epoch, show=True, path=img_path)
    print("path of images: " + img_path)

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

sess.close()
