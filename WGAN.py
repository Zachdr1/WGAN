class WGAN():
    def __init__(self, dataset, lr=5e-5, c=0.01, m=64, n_critic=5, min_z=-1, max_z=1):
        # Config
        self.lr = lr
        self.c = c
        self.m = m
        self.n_critic = 5
        self.min_z = min_z
        self.max_z = max_z
        
        # Models
        self.C = self.critic()
        self.G = self.generator()
        
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
        return -tf.reduce_mean(C_fake)
    
    @tf.function
    def C_grad(self, real_inp, fake_inp):
        with tf.GradientTape() as tape:
            C_real = self.C(real_inp, training=True)
            C_fake = self.C(fake_inp, training=True)
            loss = self.C_loss(C_real, C_fake)
        return tape.gradient(loss, self.C.trainable_variables)
    
    @tf.function
    def G_grad(self, z):
        with tf.GradientTape() as C_tape, tf.GradientTape() as G_tape:
            fake = self.G(z, training=True)
            C_fake = self.C(fake, training=False)
            loss = self.G_loss(C_fake)
        return C_tape.gradient(loss, self.G.trainable_variables)
    
    def clip_weights(self):
        for l in self.C.layers:
            weights = l.get_weights()
            weights = [tf.clip_by_value(w, -self.c, self.c) for w in weights]
            l.set_weights(weights)
    
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
    
    def critic(self):
        dropout_prob = .4

        inputs = Input(shape=(32, 32, 3))

        # Input size = 32x32x3
        x = Conv2D(filters=128, kernel_size=5, padding='same', strides=(2, 2), input_shape=(32, 32, 3))(inputs)
        x = LeakyReLU(0.2)(x)
        # Output size = 16x16x128

        # Input size = 16x16x128
        x = Conv2D(filters=256, kernel_size=5, padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # Output size = 8x8x256

        # Input size = 8x8x256
        x = Conv2D(filters=512, kernel_size=5, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # Output size = 4x4x512

        # Input size = 4x4x512
        x = Conv2D(filters=1024, kernel_size=5, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # Output size = 4x4x1024

        # Input size = 4x4x1024
        x = Flatten()(x)
        out = Dense(1)(x)

        net = Model(inputs=inputs, outputs=out)

        return net
    
    def generator(self):
        dropout_prob = .4

        # Input size = 100
        inputs = Input(shape=(100,))
        x = Dense(4*4*1024, input_shape=(100,))(inputs)
        x = Reshape(target_shape=(4, 4, 1024))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(dropout_prob)(x)
        # Output size = 4x4x1024

        # Input size = 4x4x1024
        x = Conv2DTranspose(filters=512, kernel_size=5, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(dropout_prob)(x)
        # Output size = 8x8x512

        # Input size = 8x8x512
        x = Conv2DTranspose(filters=256, kernel_size=5, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(dropout_prob)(x)
        # Output size = 16x16x256

        # Input size = 16x16x256
        x = Conv2DTranspose(filters=128, kernel_size=5, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(dropout_prob)(x)
        # Output size = 32x32x128

        # Input size = 32x32x128
        x = Conv2DTranspose(filters=3, kernel_size=5, padding='same')(x)
        out = Activation('tanh')(x)
        # Output size = 32x32x3

        net = Model(inputs=inputs, outputs=out)

        return net
        