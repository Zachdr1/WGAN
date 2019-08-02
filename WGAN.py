import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv2D,
    LeakyReLU,
    Dropout,
    BatchNormalization,
    Flatten,
    Dense,
    Activation,
    Input,
    Reshape,
    Conv2DTranspose,
    UpSampling2D
)
import numpy as np
import datetime
import os
from IPython.display import clear_output
import matplotlib.pyplot as plt


class WGAN():
    def __init__(self, dataset, lr=5e-5, c=0.01, m=64, n_critic=5, min_z=-1.0, max_z=1.0, step=0):
        # Config
        self.lr = lr
        self.c = c
        self.m = m
        self.n_critic = n_critic
        self.min_z = min_z
        self.max_z = max_z
        
        # Models
        self.C = self.critic()
        self.G = self.generator()

        # Losses
        # self.C_loss_ = None
        # self.G_loss_ = None
        self.C_loss_ = tf.keras.metrics.Mean('C_loss', dtype=tf.float32)
        self.G_loss_ = tf.keras.metrics.Mean('G_loss', dtype=tf.float32)

        # Data
        self.dataset = dataset
        self.data_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
        
        # Optimizers
        self.C_opt = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        self.G_opt = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

    @tf.function
    def C_loss(self, C_real, C_fake):
        return tf.reduce_mean(C_real) - tf.reduce_mean(C_fake)

    @tf.function
    def G_loss(self, C_fake):
        return tf.reduce_mean(C_fake)
    
    @tf.function
    def C_grad(self, real_inp, fake_inp):
        with tf.GradientTape() as tape:
            C_real = self.C(real_inp, training=True)
            C_fake = self.C(fake_inp, training=True)
            loss = self.C_loss(C_real, C_fake)
            self.C_loss_(loss)
        return tape.gradient(loss, self.C.trainable_variables)
    
    @tf.function
    def G_grad(self, z):
        with tf.GradientTape() as tape:
            fake = self.G(z, training=True)
            C_fake = self.C(fake, training=True)
            loss = self.G_loss(C_fake)
            self.G_loss_(loss)
        return tape.gradient(loss, self.G.trainable_variables)
    
    def clip_weights(self):
        for l in self.C.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -self.c, self.c) for w in weights]
            l.set_weights(weights)
    
    def clip_weight(self, w):
        return tf.clip_by_value(w, -self.c, self.c)

    @tf.function
    def C_train_on_batch(self):
        z = tf.random.uniform((self.m, 100,), self.min_z, self.max_z)
        fake_inp = self.G(z, training=False)
        real_inp = self.get_data_batch()
        grad = self.C_grad(real_inp, fake_inp)
        self.C_opt.apply_gradients(zip(grad, self.C.trainable_variables))
    
    @tf.function
    def G_train_on_batch(self):
        z = tf.random.uniform((self.m, 100,), self.min_z, self.max_z)
        grad = self.G_grad(z)
        self.G_opt.apply_gradients(zip(grad, self.G.trainable_variables))

    @tf.function
    def get_data_batch(self):
        return self.data_iter.get_next()

    def C_load(self, fp):
        self.C.load_weights(fp)
    
    def G_load(self, fp):
        self.G.load_weights(fp)
    
    # def train(self, steps=100):
    #     for step in range(0, steps):
    #         for i in range(0,5):
    #             self.C_train_on_batch()
    #             self.clip_weights()
    #         self.G_train_on_batch()
    #         tf.summary.scalar('C_loss', self.C_loss_, step=step)
    #         tf.summary.scalar('G_loss', self.G_loss_, step=step)
                
    #         if step % 50 == 0:
    #             clear_output(wait=True)
    #             noise = np.random.uniform(-1.0, 1.0, size=(5,100,))
    #             fake_batch = self.G(noise, training=False)
    #             fig = plt.figure()
    #             plt.axis('off')
    #             fake_batch = (fake_batch + 1) / 2
    #             plt.imshow((fake_batch[1]))
    #             plt.savefig(f'{image_dir}/{step}.png')
                
    #         print(f"[Step: {step}]")

    def critic(self):
        dropout_prob = .4

        inputs = Input(shape=(128, 128, 3))

        # Input size = 128x128x3
        x = Conv2D(filters=128, kernel_size=5, padding='same', strides=(2, 2), use_bias=False)(inputs)
        x = LeakyReLU(0.02)(x)
        # Output size = 64x64x128

        # Input size = 64x64x128
        x = Conv2D(filters=128, kernel_size=5, padding='same', strides=(2, 2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 32x32x256

        # Input size = 32x32x128
        x = Conv2D(filters=256, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 32x32x256

        # Input size = 32x32x128
        x = Conv2D(filters=256, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 32x32x256

        # Input size = 32x32x128
        x = Conv2D(filters=256, kernel_size=5, padding='same', strides=(2,2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 16x16x256

        # Input size = 16x16x128
        x = Conv2D(filters=256, kernel_size=5, padding='same', strides=(2, 2), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 8x8x256

        # Input size = 8x8x256
        x = Conv2D(filters=512, kernel_size=5, strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 4x4x512

        # Input size = 4x4x512
        x = Conv2D(filters=1024, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 4x4x1024

        # Input size = 4x4x1024
        x = Flatten()(x)
        out = Dense(1)(x)

        net = Model(inputs=inputs, outputs=out)

        return net
    
    def generator(self):
        # Input size = 100
        inputs = Input(shape=(100,))
        x = Dense(4*4*1024, input_shape=(100,))(inputs)
        x = Reshape(target_shape=(4, 4, 1024))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 4x4x1024

        # Input size = 4x4x1024
        x = Conv2D(filters=512, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        x = UpSampling2D()(x)
        # Output size = 8x8x512

        # Input size = 8x8x512
        x = Conv2D(filters=256, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        x = UpSampling2D()(x)
        # Output size = 16x16x256

        # Input size = 16x16x512
        x = Conv2D(filters=256, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 16x16x256

        # Input size = 16x16x256
        x = Conv2D(filters=128, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        x = UpSampling2D()(x)

        # Output size = 32x32x128

        # Input size = 32x32x256
        x = Conv2D(filters=128, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        x = UpSampling2D()(x)
        # Output size = 64x64x128

        # Input size = 64x64x256
        x = Conv2D(filters=128, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        x = UpSampling2D()(x)
        # Output size = 128x128x128

        # Input size = 128x128x256
        x = Conv2D(filters=128, kernel_size=5, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.02)(x)
        # Output size = 128x128x128


        # Input size = 128x128x128
        x = Conv2D(filters=3, kernel_size=5, padding='same', use_bias=False)(x)
        out = Activation('tanh')(x)
        # Output size = 32x32x3

        net = Model(inputs=inputs, outputs=out)
        
        return net
        