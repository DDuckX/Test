import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
from time import strftime
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score
import os
import time
import multiprocessing
from multiprocessing import Pool


dec_data = np.loadtxt(r'G:\BenchmarkDatasets\NoAuction\1.NoAuction_Zscore\NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')
dec_train = dec_data[:, :int(dec_data.shape[1] * 0.8)]
dec_val = dec_data[:, int(dec_data.shape[1] * 0.8):]
dec_test1 = np.loadtxt('G:/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')
dec_test2 = np.loadtxt('G:/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')
dec_test3 = np.loadtxt('G:/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
dec_test = np.hstack((dec_test1, dec_test2, dec_test3))



W = 40                     #number of features
dim = 40                     #number of LOB states

horizon = 2        #if h = 5 than k = 10, h = 2 then k=50
# T = 5
col=list(range(0,40))
col.append(-horizon)
dec_train = dec_train[col, :].T
dec_val = dec_val[col, :].T
dec_test = dec_test[col, :].T
dec_train=pd.DataFrame(dec_train)
dec_test=pd.DataFrame(dec_test)
dec_val=pd.DataFrame(dec_val)

def transdataclass(dataset):
    dataset['up']=0
    dataset['sta']=0
    dataset['down']=0
    dataset.loc[dataset[40]==1,'up']=1
    dataset.loc[dataset[40] == 2, 'sta'] = 1
    dataset.loc[dataset[40] == 3, 'down'] = 1
    dataset.drop(columns=[40], inplace=True)
    return dataset


def to_windows(dataset, seq_len,batchsize,shift,batch=False,shuffle=False):
    dataset=tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(seq_len, shift=shift, drop_remainder=True)
    dataset=dataset.flat_map(lambda window_ds: window_ds.batch(seq_len))
    dataset=dataset.map(lambda window: (window[:,:-3], window[-1,-3:],))
    if shuffle:
        dataset=dataset.shuffle(5*batchsize,seed=11)
    if batch:
        dataset = dataset.batch(batchsize)
    else:
        dataset=dataset.batch(1)
    return dataset

def get_run_logdir(root_logdir=r"mylogs"):
    return Path(root_logdir)/strftime("run_%Y_%m_%d_%H_%M_%S")

def inception_module(layer_in, f1_in,f1_out,f2_in , f2_out, f3_in, f3_out):
  leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
  # 3x1 conv
  conv3 = tf.keras.layers.Conv2D(f1_in, (1,1), padding='same', activation=leaky_relu)(layer_in)
  conv3= tf.keras.layers.Conv2D(f1_out, (3,1), padding='same', activation=leaky_relu)(conv3)
  # 5x1 conv
  conv5 = tf.keras.layers.Conv2D(f2_in, (1,1), padding='same', activation=leaky_relu)(layer_in)
  conv5 = tf.keras.layers.Conv2D(f2_out, (5,1), padding='same', activation=leaky_relu)(conv5)

  # 3x3 max pooling
  pool =tf.keras.layers.MaxPooling2D((3,1),strides=(1,1),padding='same')(layer_in)
  pool =tf.keras.layers.Conv2D(f3_out, (1,1), padding='same', activation=leaky_relu)(pool)
  # concatenate filters
  layer_out =  tf.keras.layers.concatenate([conv3, conv5, pool], axis=-1)
  return layer_out

def generator_model():
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
    inputs = tf.keras.Input(shape=(128, 40, 1))
    conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 2), strides=(1, 2), activation=leaky_relu)(inputs)
    conv11 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), padding='same', activation=leaky_relu)(conv1)
    conv12 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), padding='same', activation=leaky_relu)(conv11)
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 2), strides=(1, 2), activation=leaky_relu)(conv12)
    conv21 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), padding='same', activation=leaky_relu)(conv2)
    conv22 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), padding='same', activation=leaky_relu)(conv21)
    conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 10), activation=leaky_relu)(conv22)
    conv31 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), padding='same', activation=leaky_relu)(conv3)
    conv32 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), padding='same')(conv31)
    inception = inception_module(conv32, f1_in=32, f1_out=32, f2_in=32, f2_out=32, f3_in=32, f3_out=32)
    reshape = tf.keras.layers.Reshape((128, 96))(inception)
    lstm = tf.keras.layers.LSTM(64)(reshape)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(lstm)
    model=tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def discriminator_model():
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=4,strides=2, activation=leaky_relu, input_shape=(128,43)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=2, activation=leaky_relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=4, strides=2, activation=leaky_relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=leaky_relu))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def gen_optimizator(name,lr):
    if name=='adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name=='sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        raise ValueError('Invalid optimizer name: {}'.format(name))

def dis_optimizator(name,lr):
    if name=='adam':
        return tf.keras.optimizers.Adam(learning_rate=lr*100)
    elif name=='sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr*100)
    else:
        raise ValueError('Invalid optimizer name: {}'.format(name))

class GAN:
    def __init__(self,generator, discriminator,gen_metrics,dis_metrics,lr,genopt,disopt,log_dir):
        self.generator = generator
        self.discriminator = discriminator
        self.lr=lr
        self.genopt=genopt
        self.disopt=disopt
        self.gen_opt=gen_optimizator(self.genopt,self.lr)
        self.dis_opt=dis_optimizator(self.disopt,self.lr)
        self.checkpoint_dir = 'GAN/training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(gen_opt=self.gen_opt,dis_opt=self.dis_opt,generator=self.generator,discriminator=self.discriminator
                                              )
        self.generator_metrics = gen_metrics
        self.discriminator_metrics = dis_metrics
        self.train_summary = tf.summary.create_file_writer(log_dir)
        self.loss_mean=tf.keras.metrics.Mean()


    def discriminator_loss(self,real_out, fake_out):
        real_loss_obj = tf.keras.losses.BinaryCrossentropy()
        real_loss=real_loss_obj(tf.ones_like(real_out), real_out)
        fake_loss_obj = tf.keras.losses.BinaryCrossentropy()
        fake_loss=fake_loss_obj(tf.zeros_like(fake_out), fake_out)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,y_true, y_pred, fake_out):
        generate_loss_obj=tf.keras.losses.CategoricalCrossentropy()
        generate_loss = generate_loss_obj(y_true, y_pred)
        discrminator_loss_obj = tf.keras.losses.BinaryCrossentropy()
        discrminator_loss=discrminator_loss_obj(tf.ones_like(fake_out), fake_out)
        total_loss = generate_loss + discrminator_loss
        return total_loss


    @tf.function
    def train_process(self, trainset, seq_len,epoch):
        G_losses = []
        D_losses = []
        gtrain_loss = self.loss_mean
        dtrain_loss= self.loss_mean
        G_loss =0.
        D_loss=0.
        y_pred = tf.zeros(shape=(1,))
        for X,Y in trainset:
            X = tf.cast(X, tf.float32)
            Y = tf.cast(Y, tf.float32)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                y_pred = self.generator(tf.expand_dims(X,axis=-1), training=True)
                fake_in = tf.tile(tf.expand_dims(y_pred, axis=1), [1, seq_len, 1])
                fake_in = tf.concat([X, fake_in], axis=-1)
                real_in = tf.tile(tf.expand_dims(Y, axis=1), [1, seq_len, 1])
                real_in = tf.concat([X, real_in], axis=-1)

                fake_out = self.discriminator(fake_in, training=True)
                real_out = self.discriminator(real_in, training=True)

                gen_loss = self.generator_loss(Y, y_pred, fake_out)
                dis_loss = self.discriminator_loss(real_out, fake_out)

            gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            for m in  self.generator_metrics:
                m.update_state(Y, y_pred)

            dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
            self.dis_opt.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))
            for m in self.discriminator_metrics:
                m.update_state(tf.concat([tf.ones_like(fake_out), tf.zeros_like(real_out)], axis=0),
                          tf.concat([fake_out, real_out], axis=0))

            G_loss=gtrain_loss(gen_loss)
            D_loss=dtrain_loss(dis_loss)

        tf.print('epoch:', epoch + 1, 'g_loss', G_loss, 'd_loss', D_loss,
                 "G_Accuracy:", self.generator_metrics[0].result(),
                 "G_Precision:", self.generator_metrics[1].result(),
                 "G_Recall:", self.generator_metrics[2].result(),
                 "D_Accuracy:", self.discriminator_metrics[0].result())
        G_losses.append(G_loss)
        D_losses.append(D_loss)
        with self.train_summary.as_default():
            tf.summary.scalar('g_loss', G_loss, step=epoch)
            tf.summary.scalar('d_loss', D_loss, step=epoch)
            for metrics in self.generator_metrics:
                name=metrics.name
                value=metrics.result()
                tf.summary.scalar(name, value, step=epoch)
        gtrain_loss.reset_state()
        dtrain_loss.reset_state()
        for m in self.generator_metrics:
            m.reset_state()
        for m in self.discriminator_metrics:
            m.reset_state()
        return y_pred, {'g_loss':G_losses,'d_loss': D_losses},D_losses,G_losses

    @tf.function
    def test_process(self, valiset, seq_len,epoch):
        Vali_G_losses = []
        Vali_D_losses = []
        prediction=[]
        Vgtrain_loss = self.loss_mean
        Vdtrain_loss= self.loss_mean
        Vali_G_loss = 0.
        Vali_D_loss = 0.
        generator_metrics={}
        y_pred = tf.zeros(shape=(1,))
        for vali_X,vali_Y in valiset:
            vali_X = tf.cast(vali_X, tf.float32)
            vali_Y = tf.cast(vali_Y, tf.float32)
            y_pred = self.generator(vali_X, training=False)
            fake_in = tf.tile(tf.expand_dims(y_pred, axis=1), [1, seq_len, 1])
            fake_in = tf.concat([vali_X, fake_in], axis=-1)
            real_in = tf.tile(tf.expand_dims(vali_Y, axis=1), [1, seq_len, 1])
            real_in = tf.concat([vali_X, real_in], axis=-1)

            fake_out = self.discriminator(fake_in, training=False)
            real_out = self.discriminator(real_in, training=False)

            gen_loss = self.generator_loss(vali_Y, y_pred, fake_out)
            dis_loss = self.discriminator_loss(real_out, fake_out)
            for m in self.generator_metrics:
                m.update_state(vali_Y, y_pred)
            for m in self.discriminator_metrics:
                m.update_state(tf.concat([tf.ones_like(fake_out), tf.zeros_like(real_out)], axis=0),
                          tf.concat([fake_out, real_out], axis=0))
            Vali_G_loss=Vgtrain_loss(gen_loss)
            Vali_D_loss=Vdtrain_loss(dis_loss)

        tf.print('epoch:', epoch + 1, 'Vali_g_loss', Vali_G_loss, 'Vali_d_loss', Vali_D_loss,
                 "G_Accuracy:", self.generator_metrics[0].result(),
                 "G_Precision:", self.generator_metrics[1].result(),
                 "G_Recall:", self.generator_metrics[2].result(),
                 "D_Accuracy:", self.discriminator_metrics[0].result())

        Vali_D_losses.append(Vali_G_loss)
        Vali_G_losses.append(Vali_G_loss)
        prediction.append(y_pred)
        with self.train_summary.as_default():
            tf.summary.scalar('Vali_g_loss',  Vali_G_loss, step=epoch)
            tf.summary.scalar('Vali_d_loss',  Vali_D_loss, step=epoch)
            for metrics in self.generator_metrics:
                name='Vali_'+metrics.name
                value=metrics.result()
                tf.summary.scalar(name, value, step=epoch)
        Vgtrain_loss.reset_state()
        Vdtrain_loss.reset_state()

        for metrics in self.generator_metrics:
            name = metrics.name
            value = metrics.result()
            if name not in generator_metrics:
                generator_metrics[name] = []
            generator_metrics[name].append(value)
        for m in self.generator_metrics:
            m.reset_state()
        for m in self.discriminator_metrics:
            m.reset_state()

        return y_pred, {'g_loss': Vali_G_loss, 'd_loss': Vali_D_loss}, generator_metrics,prediction

    def train(self, trainset, valiset, seq_len, epochs):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        best_score = 0

        for epoch in range(epochs):
            starttime = time.time()
            pre_pf, loss,D_losses,G_losses = self.train_process(trainset, seq_len,epoch)

            vali_pre, vali_loss, vali_metrics,prediction = self.test_process(valiset, seq_len,epoch)

        if (epoch+1)%10==0:
            tf.keras.models.save_model(self.generator, 'GAN/gen_model_%d.h5' % epoch)
            self.checkpoint.save(file_prefix=self.checkpoint_prefix + f'-{epoch}')



            if vali_metrics['precision'][-1] > best_score:
                best_score = vali_metrics['precision'][-1]
                tf.keras.models.save_model(self.generator, 'GAN/best_generator.h5')
                tf.keras.models.save_model(self.discriminator, 'GAN/best_discriminator.h5')
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - starttime))



            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)

        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_GAN.png')
        prediction.to_csv('../GAN/predction k=50.csv')
    def compile(self):
        self.generator.compile(
            metrics=self.generator_metrics
        )
        self.discriminator.compile(
            metrics=self.discriminator_metrics
        )
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)
seq_len=int(128)
batchsize=int(64)
dec_train=transdataclass(dec_train)
dec_val=transdataclass(dec_val)
dec_test=transdataclass(dec_test)
train_ds = to_windows(dec_train, seq_len, batchsize, shift=1, batch=True)
valid_ds = to_windows(dec_val, seq_len, batchsize, shift=1, batch=True)
test_ds = to_windows(dec_test, seq_len, batchsize=1, shift=1)
generator_metrics = [tf.keras.metrics.CategoricalAccuracy(),
                     # F1Score(num_classes=3,average='macro'),
                     tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
discriminator_metrics = [tf.keras.metrics.Accuracy()]
generator=generator_model()
discriminator=discriminator_model()
log_dir=str(get_run_logdir())

gan=GAN(generator, discriminator,gen_metrics=generator_metrics,dis_metrics=discriminator_metrics,lr=0.001,genopt='adam',disopt='adam',log_dir=log_dir)
gan.compile()
gan.train(train_ds,valid_ds,seq_len,100)



