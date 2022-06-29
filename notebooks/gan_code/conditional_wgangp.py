import numpy as np

import os, sys
import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from functools import partial
tf.keras.backend.set_floatx('float32')

sys.path.append('gan_code/')
import DataLoader 
import importlib
importlib.reload(DataLoader)

class WGANGP():
  def __init__(self):
    dgratio = 5
    batchsize = 128
    G_lr = D_lr = 0.0001
    G_beta1 = D_beta1 = 0.55
    self.lam = 0.0001

    # Construct D and G models
    self.G = self.make_generator_functional_model() 
    self.D = self.make_discriminator_model()

    # Construct D and G optimizers
    self.generator_optimizer = tf.optimizers.Adam(learning_rate=G_lr, beta_1=G_beta1)
    self.discriminator_optimizer = tf.optimizers.Adam(learning_rate=D_lr, beta_1=D_beta1)

    # Prepare for check pointing
    self.saver = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                               discriminator_optimizer=self.discriminator_optimizer,
                               generator=self.G,
                               discriminator=self.D)

    self.tf_batchsize = tf.constant(batchsize, dtype=tf.int32)
    self.tf_dgratio = tf.constant(dgratio, dtype=tf.int32)
 
  def make_generator_functional_model(self):
    initializer = tf.keras.initializers.he_uniform()
    bias_node = True
    noise = layers.Input(shape=(50), name="Noise")
    condition = layers.Input(shape=(2), name="mycond")
    con = layers.concatenate([noise,condition])
    G = layers.Dense(50, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(con)  
    G = layers.BatchNormalization()(G)
    G = layers.Activation(activations.swish)(G)
    G = layers.Dense(100, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(G)
    G = layers.BatchNormalization()(G)
    G = layers.Activation(activations.swish)(G)
    G = layers.Dense(200, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(G)
    G = layers.BatchNormalization()(G)
    G = layers.Activation(activations.swish)(G)
    G = layers.Dense(368, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(G)
    G = layers.BatchNormalization()(G)
    G = layers.Activation(activations.swish)(G)

    generator = Model(inputs=[noise, condition], outputs=G)
    generator.build(370)
    generator.summary()
    return generator

  def make_discriminator_model(self):
    initializer = tf.keras.initializers.he_uniform()
    bias_node = True

    image = layers.Input(shape=(368), name="Image")
    d_condition = layers.Input(shape=(2), name="mycond")
    d_con = layers.concatenate([image,d_condition])
    D = layers.Dense(368, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(d_con)  
    D = layers.Activation(activations.relu)(D)
    D = layers.Dense(368, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(D)
    D = layers.Activation(activations.relu)(D)
    D = layers.Dense(368, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(D)
    D = layers.Activation(activations.relu)(D)
    D = layers.Dense(1, use_bias=bias_node, kernel_initializer=initializer, bias_initializer='zeros')(D)

    discriminator = Model(inputs=[image, d_condition], outputs=D)
    discriminator.build(370)
    discriminator.summary()
    return discriminator

  @tf.function
  def gradient_penalty(self, f, x_real, x_fake, cond_label):
    alpha = tf.random.uniform([self.tf_batchsize, 1], minval=0., maxval=1.)

    inter = alpha * x_real + (1-alpha) * x_fake
    with tf.GradientTape() as t:
      t.watch(inter)
      pred = self.D(inputs=[inter, cond_label])
    grad = t.gradient(pred, [inter])[0]
    
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
    gp = self.lam * tf.reduce_mean((slopes - 1.)**2)
    return gp

  @tf.function
  def D_loss(self, x_real, cond_label): 
    z = tf.random.normal([self.tf_batchsize, 50], mean=0.5, stddev=0.5, dtype=tf.dtypes.float32)
    x_fake = self.G(inputs=[z, cond_label])
    D_fake = self.D(inputs=[x_fake, cond_label])
    D_real = self.D(inputs=[x_real, cond_label])
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + self.gradient_penalty(f = partial(self.D, training=True), x_real = x_real, x_fake = x_fake, cond_label=cond_label)
    return D_loss, D_fake

  @tf.function
  def G_loss(self, D_fake):
    G_loss = -tf.reduce_mean(D_fake)
    return G_loss

  def getTrainData_ultimate(self, n_epoch):
    true_batchsize = tf.cast(tf.math.multiply(self.tf_batchsize, self.tf_dgratio), tf.int64)
    n_samples = tf.cast(tf.gather(tf.shape(self.X), 0), tf.int64)
    n_batch = tf.cast(tf.math.floordiv(n_samples, true_batchsize), tf.int64)
    n_shuffles = tf.cast(tf.math.ceil(tf.divide(n_epoch, n_batch)), tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((self.X, self.Labels))
    ds = ds.shuffle(buffer_size = n_samples).repeat(n_shuffles).batch(true_batchsize, drop_remainder=True).prefetch(2)
    self.ds = ds
    self.ds_iter = iter(ds)
    X_feature_size = tf.gather(tf.shape(self.X), 1)
    Labels_feature_size = tf.gather(tf.shape(self.Labels), 1)
    self.X_batch_shape = tf.stack((self.tf_dgratio, self.tf_batchsize, X_feature_size), axis=0)
    self.Labels_batch_shape = tf.stack((self.tf_dgratio, self.tf_batchsize, Labels_feature_size), axis=0)

  @tf.function
  def train_loop(self, X_trains, cond_labels): 
    for i in tf.range(self.tf_dgratio):
      with tf.GradientTape() as disc_tape:
        (D_loss_curr, D_fake) = self.D_loss(tf.gather(X_trains, i), tf.gather(cond_labels, i))
        gradients_of_discriminator = disc_tape.gradient(D_loss_curr, self.D.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

    last_index = tf.subtract(self.tf_dgratio, 1)
    with tf.GradientTape() as gen_tape:
      # Need to recompute D_fake, otherwise gen_tape doesn't know the history
      (D_loss_curr, D_fake) = self.D_loss(tf.gather(X_trains, last_index), tf.gather(cond_labels, last_index))
      G_loss_curr = self.G_loss(D_fake)
      gradients_of_generator = gen_tape.gradient(G_loss_curr, self.G.trainable_variables)
      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
      return D_loss_curr, G_loss_curr


  def train(self):
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    print ('training started')
    dl = DataLoader.DataLoader()

    start_iteration = 0 
    max_iterations = 1000

    for iteration in range(start_iteration,max_iterations): 
        if (iteration == 0):
            X, Labels = dl.getAllTrainData(8, 9)
            self.X = tf.convert_to_tensor(X, dtype=tf.float32)
            self.Labels = tf.convert_to_tensor(Labels, dtype=tf.float32)
        
            remained_iteration = tf.constant(max_iterations - iteration, dtype=tf.int64)
            self.getTrainData_ultimate(remained_iteration)
            print ("Using "+ str(self.X.shape[0])+ " events")

        X, Labels = self.ds_iter.get_next()
        X_trains    = tf.reshape(X, self.X_batch_shape)
        cond_labels = tf.reshape(Labels, self.Labels_batch_shape)   
        D_loss_curr, G_loss_curr = self.train_loop(X_trains, cond_labels)

        if iteration == 0: 
            print("Model and loss values will be saved every 2 iterations." )
        
        if iteration % 2 == 0 and iteration > 0:

            try:
                self.saver.save(file_prefix = checkpoint_dir+ '/model')
            except:
                print("Something went wrong in saving iteration %s, moving to next one" % (iteration))
                print("exception message ", sys.exc_info()[0])     
                
            print('Iter: {}; D loss: {:.4}; G_loss:  {:.4}'.format(iteration, D_loss_curr, G_loss_curr))
    
        

  def load(self, epoch, labels, nevents, input_dir_gan):
    checkPointName = "%s/model-%s" % (input_dir_gan, int(epoch))
    print(checkPointName)
    self.saver.restore(checkPointName)

    z = tf.random.normal([nevents, 50], mean=0.5, stddev=0.5, dtype=tf.dtypes.float32)
    x_fake = self.G(inputs=[z, labels])
    return x_fake

  def SaveModelForLWTNN(self, checkpointfile, output_dir): 
    print("Saving model from %s in %s" % (checkpointfile, output_dir))
    self.saver.restore(checkpointfile)
    
    self.G.save_weights(output_dir + '/checkpoint_%s_eta_%s_%s.h5' % (self.voxInputs.particle, self.voxInputs.eta_min, self.voxInputs.eta_max))
    generator_model_json = self.G.to_json()
    with open(output_dir + "/generator_model_%s_eta_%s_%s.json" % (self.voxInputs.particle, self.voxInputs.eta_min, self.voxInputs.eta_max), "w") as json_file:
      json_file.write(generator_model_json)    

