import tensorflow as tf
from WGAN import WGAN

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

(x_train, _), (_,_) = tf.keras.datasets.cifar10.load_data()

def preprocess_fn(image):
    x = tf.reshape(tf.cast(image, tf.float32), (32,32,3))
#     x /= 255
    x = 2*x/255 - 1 # convert image to [-1, 1] range
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

real_ds = tf.data.Dataset.from_tensor_slices(x_train)
real_ds = real_ds.shuffle(60000)
real_ds = real_ds.repeat()
real_ds = real_ds.apply(tf.data.experimental.map_and_batch(
        preprocess_fn, 64, num_parallel_batches=6, drop_remainder=True))
real_ds = real_ds.prefetch(tf.data.experimental.AUTOTUNE)

wgan = WGAN(real_ds)

for step in range(0, 15001):
    for i in range(0, 5):
        wgan.C_train_on_batch()
        wgan.clip_weights()
    wgan.G_train_on_batch()
        
    if step % 50 == 0:
        pass
    print(f"Step: {step}")

import numpy as np
import random
import matplotlib.pyplot as plt
noise = np.random.uniform(-1.0, 1.0, size=(5,100,))
fake_batch = wgan.G(noise)
print(fake_batch.shape)

fig = plt.figure()
plt.axis('off')
fake_batch = (fake_batch + 1) / 2
plt.imshow((fake_batch[3]))