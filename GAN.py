import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import time
import pandas as pd
from IPython import display

#Creating the generator model for GAN -> Generator and Discriminator 
#it's a sequential model -> doesn't need the flat layer at the end 
def make_generator_model():
  #7*7*256 units at this layer with input shape is an array from 0-100
    model = tf.keras.Sequential()
    model.add(layers.Dense(20*20*256, use_bias=False, input_shape=(100,)))#this is the input shape for the noise 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #checking the shape of the output 
    model.add(layers.Reshape((20, 20, 256)))#just check the shape 
    assert model.output_shape == (None, 20, 20, 256) # Note: None is the batch size
  #changing the shape of the output 
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 20, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
#changing the shape of the output 
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 40, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 80, 80, 1)#the output is like the image 

    return model


#instanciate the generator model 
generator = make_generator_model()

noise = tf.random.normal([1, 100])#fit some random noise vector 
#noise is the input to our model 
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0],cmap='gray')

#Discriminator model -> simple classifier 
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[80, 80, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))#reduce overfitting 

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # wx+b

    return model

#instanciate the discriminator model 
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)#for classification task(binary)

def discriminator_loss(real_output, fake_output):
  #we are acting on the outputs 
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)#if is real is expected to be 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) #fake output [1,1,1,1,1], gth [0,0,0,0,0]
    #insteadd of taking tf.ones could be better to take 0.9  or the discriminator will learn too fast 
    #if is fake is expected to be 0 
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)#fake output [1,1,1,1,1], gth [1,1,1,1,1]

#optimizers for the models 
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 4000
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])#seed for the generator 

#define the custom train function 
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])#creating the noise 

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:#two gradiinet tapes for the model 
      generated_images = generator(noise, training=True)#fit noise on the generator 
      #training=true means thta your model will be back propagated for training 
  #if u are doing validation u can set on false 
      real_output = discriminator(images, training=True)#getting the output form real images on discriminator 
      fake_output = discriminator(generated_images, training=True)#getting the output of fake images on discriminator 
    #calculate losses 
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
  #get the gradient wrt the loss from generator and discriminator 
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#optimize the models wrt the gradients (gradient descent step)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#this is the train function -> start time , safe checkpoint print some information 
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  #fig = plt.figure(figsize=(4,4))

  #for i in range(predictions.shape[0]):
  #plt.subplot(4, 4, i+1)
  plt.imshow(predictions[1, :, :, 0] * 127.5 + 127.5,cmap='gray')
      #plt.axis('off')

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

#if u want to save parameters 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

