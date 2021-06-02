import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import scipy.io as sio

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)






# def load_data(in_dt,out_dt,mode='train'):
#     # 数据集划分
#     if mode == 'train': #75%
#         in_dt = in_dt[:int(0.75 * len(in_dt))]
#         out_dt = out_dt[:int(0.75 * len(out_dt))]
#     elif mode == 'val': # %
#         in_dt = in_dt[int(0.75 * len(in_dt)):int(0.85 * len(in_dt))]
#         out_dt = out_dt[int(0.75 * len(out_dt)):int(0.85 * len(out_dt))]
#     else: # 10%
#         in_dt = in_dt[int(0.85 * len(in_dt)):]
#         out_dt = out_dt[int(0.85 * len(out_dt)):]
#     return in_dt, out_dt
# in_dt, out_dt = load_data(in_dt,out_dt,mode='train')
# train_dataset = tf.data.Dataset.from_tensor_slices((in_dt, out_dt))
# in_dt2, out_dt2 = load_data(in_dt,out_dt,mode='val')
# val_dataset = tf.data.Dataset.from_tensor_slices((in_dt2, out_dt2))
# in_dt3,out_dt3 = load_data(in_dt,out_dt,mode='test')
# test_dataset = tf.data.Dataset.from_tensor_slices((in_dt3, out_dt3))



# data_z = np.empty((inputLength, 1))
# in_da = np.empty((175, 1024, 1))
# in_da1 = np.empty((1, 1024, 20))
# for root, dirs, files in os.walk(file_path):
#     for file in files:
#         if os.path.splitext(file)[1] == '.txt':
#             data1 = np.loadtxt(root + '\\' + file)
#             for i in range(20):
#                 d1 = data1[:, i]
#                 d1 = d1.reshape(180000, 1)
#                 for i in range(176):
#                     dt1 = d1[i * 1024:(i * 1024 + inputLength), :]
#                     if dt1.shape[0] == inputLength:
#                         data_z = np.append(data_z, dt1, axis=1)  # 1024*176
#                 data_z = np.delete(data_z, 0, axis=1)
#                 data_z = data_z.T
#                 # data_z=data_z.transpose(())
#                 data_z = data_z.reshape(data_z.shape[0], data_z.shape[1], 1)
#                 in_da = np.concatenate((in_da, data_z), axis=2)
#                 data_z = np.empty((inputLength, 1))
#             in_da = in_da[:, :, 1:]
#             in_da1 = np.concatenate((in_da1, in_da), axis=0)
#             in_da = np.empty((175, 1024, 1))
# in_da1 = in_da1[1:, :, :]
# in_dt = in_da1[:, :, :-1]
# out_dt = in_da1[:, :, -1]
# out_dt = out_dt.reshape((out_dt.shape[0], out_dt.shape[1], 1))

#
# in_d1 = np.empty((1, 1024, 20))
# in_d = np.empty((175, 1024, 1))
# for root, dirs, files in os.walk(r'G:\Nonlinear model\text\input\val'):
#     for file in files:
#         if os.path.splitext(file)[1] == '.txt':
#             data2 = np.loadtxt(root + '\\' + file)
#             for i in range(20):
#                 d2 = data2[:, i]
#                 d2 = d2.reshape(180000, 1)
#                 for i in range(176):
#                     dt2 = d2[i * 1024:(i * 1024 + inputLength), :]
#                     if dt2.shape[0] == inputLength:
#                         data_z = np.append(data_z, dt2, axis=1)  # 1024*176
#                 data_z = np.delete(data_z, 0, axis=1)
#                 data_z = data_z.T
#                 # data_z=data_z.transpose(())
#                 data_z = data_z.reshape(data_z.shape[0], data_z.shape[1], 1)
#                 in_d = np.concatenate((in_d, data_z), axis=2)
#                 data_z = np.empty((inputLength, 1))
#             in_d = in_d[:, :, 1:]
#             in_d1 = np.concatenate((in_d1, in_d), axis=0)
#             in_d = np.empty((175, 1024, 1))
# in_d1 = in_d1[1:, :, :]
# in_dv = in_d1[:, :, :-1]
# out_dv = in_d1[:, :, -1]
# out_dv = out_dv.reshape((out_dv.shape[0], out_dv.shape[1], 1))
BATCH_SIZE = 8
inputLength = 1024
EPOCHS = 1
is_training = True
sampledistance = 1024

file_path1 = r'G:\gzhu\Guangzhou TV\train.mat'
file_path2 = r'G:\gzhu\Guangzhou TV\val.mat'


def load_mat_train(mat_path):
    # load training data
    f = sio.loadmat(mat_path)
    X_train = np.array(f['train_input'])
    Y_train = np.array(f['train_output'])
    dim_X = X_train.shape
    if len(dim_X) == 2:
        X_train = X_train.reshape(dim_X[0], dim_X[1], 1)
    dim_Y = Y_train.shape
    if len(dim_Y) == 2:
        Y_train = Y_train.reshape(dim_Y[0], dim_Y[1], 1)

    return X_train, Y_train


def load_mat_val(mat_path):
    # load training data
    f = sio.loadmat(mat_path)
    X_val = np.array(f['val_input'])
    Y_val = np.array(f['val_output'])
    dim_X = X_val.shape
    if len(dim_X) == 2:
        X_val = X_val.reshape(dim_X[0], dim_X[1], 1)
    dim_Y = Y_val.shape
    if len(dim_Y) == 2:
        Y_val = Y_val.reshape(dim_Y[0], dim_Y[1], 1)

    return X_val, Y_val
in_dt, out_dt=load_mat_train(file_path1)
in_dv, out_dv= load_mat_val(file_path2)
in_dt = tf.convert_to_tensor(in_dt, dtype='float32')
out_dt = tf.convert_to_tensor(out_dt, dtype='float32')
in_dv = tf.convert_to_tensor(in_dv, dtype='float32')
out_dv = tf.convert_to_tensor(out_dv, dtype='float32')


# 构建模型
filter = 64
class resnet_block(tf.keras.Model):
    def __init__(self):
        super(resnet_block, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filter, 64, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.con = tf.keras.layers.Concatenate()
        self.conv2 = tf.keras.layers.Conv1D(filter, 64, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv1D(filter*2, 64, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        x = inputs
        x = tf.nn.leaky_relu(x)  # 激活函数
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.leaky_relu(self.con([inputs,x]))
        return  x


class Self_Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(Self_Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,20, input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它


    def call(self, input,**kwargs):
        x = input
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (1024 ** 0.5)

        QK = K.softmax(QK)

        V = K.batch_dot(QK, WV)

        return V


# 生成器

class Generator(tf.keras.Model):
    # 生成器网络
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filter, 128, strides=1, padding='same')

        self.conv2 = tf.keras.layers.Conv1D(filter * 2, 64, strides=1, padding='same')

        self.conv3 = tf.keras.layers.Conv1D(filter * 4, 8, strides=1, padding='same')

        self.conv4 = tf.keras.layers.Conv1D(filter*2, 4, strides=1, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv1D(1, 4, strides=1, padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.res = resnet_block()
        self.drop = tf.keras.layers.Dropout(0.5)
        self.con = tf.keras.layers.Concatenate()
        self.att = Self_Attention()
        self.conv6 = tf.keras.layers.Conv1D(1, 2, strides=1, padding='same')
        self.bn6 = tf.keras.layers.BatchNormalization()
    def call(self, inputs):
        x = inputs
        x = tf.nn.leaky_relu(x)  # 激活函数
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x)))
        x = tf.nn.leaky_relu(self.drop(x))
        x = tf.nn.leaky_relu(self.res(x))

        x = tf.nn.leaky_relu(self.bn4(self.conv4(x)))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x)))
        x = tf.nn.leaky_relu(self.drop(x))
        x = tf.nn.leaky_relu(self.res(x))

        x = tf.nn.leaky_relu(self.bn5(self.conv5(x)))
        in_attention = self.con([inputs,x])
        out_attention = self.att(in_attention)
        out_attention = out_attention[:,:,-1]
        out_attention = tf.reshape(out_attention,(-1,1024,1))
        mix_out = self.con([x,out_attention])
        x = tf.nn.leaky_relu(self.bn6(self.conv6(mix_out)))
        return x


class Discriminator(tf.keras.Model):
    # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()
        # 卷积层
        self.conv1 = tf.keras.layers.Conv1D(filter, 4, 64, 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 卷积层
        self.conv2 = tf.keras.layers.Conv1D(filter * 2, 644, 2, 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 卷积层
        self.conv3 = tf.keras.layers.Conv1D(filter * 4, 128, 2, 'same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        # 卷积层
        self.conv4 = tf.keras.layers.Conv1D(filter * 8, 64, 1,'same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        # 卷积层
        # 全局池化层
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))

        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))

        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))

        x = self.pool(x)

        x = self.fc(x)

        logits = tf.nn.sigmoid(x)

        return logits




# 定义优化器和损失函数
def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def d_loss_fn(batch_i, batch_t):
    fake_i = generator(batch_i, training=False)
    d_fake_logits = discriminator(fake_i)
    d_real_logits = discriminator(batch_t)
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss


def g_loss_fn(batch_i, y_true):
    # 在训练生成网络时，需要迫使生成图片判定为真
    fake_i = generator(batch_i)
    d_fake_logits = discriminator(fake_i, training=False)
    # 计算生成图片与1之间的误差
    loss1 = K.mean(K.sqrt(K.sum((fake_i - y_true) ** 2, axis=[1, 2])) / K.sqrt(K.sum(y_true ** 2, axis=[1, 2])))
    # loss1 = keras.losses.mse(y_pred, y_true)
    loss2 = celoss_ones(d_fake_logits)
    # loss2 = tf.constant(loss2, dtype='double')
    loss = 1*loss1 + 0.01*loss2
    return loss


# 定义优化器
generator = Generator()  # 创建生成器
generator.build(input_shape=(BATCH_SIZE,inputLength, in_dt.shape[2]))
discriminator = Discriminator()  # 创建判别器
discriminator.build(input_shape=(None,inputLength, out_dt.shape[2]))
# g_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# d_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
g_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.5)
generator.compile(optimizer=g_optimizer, loss=[vars()['g_loss_fn']])
discriminator.compile(optimizer=d_optimizer, loss=[vars()['d_loss_fn']])

d_losses, g_losses = [], []
g_losses_val=[]
for epoch in range(EPOCHS):  # 训练epochs次
    # 1. 训练判别器
    bs = BATCH_SIZE
    bn = in_dt.shape[0] // BATCH_SIZE  # total number of batch
    for i in range(bn):
        # Initialize progbar and batch counter
        # progbar = generic_utils.Progbar(X_train.shape[0])
        batch_i = in_dt[i * bs:(i + 1) * bs, :, :]
        batch_t = out_dt[i * bs:(i + 1) * bs, :, :]  # 采样真实图片
        # 判别器前向计算

        with tf.GradientTape() as tape:
            discriminator.trainable = True
            d_loss = d_loss_fn(batch_i, batch_t)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        # 2. 训练生成器
        # 生成器前向计算
        with tf.GradientTape() as tape:
            discriminator.trainable = False
            g_loss = g_loss_fn(batch_i, batch_t)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        # Create a batch to feed the discriminator model
        # x means input, y means output
        # generated_x = generator.predict(batch_i)
        # x_disc_p1 = np.concatenate([generated_x, batch_i], axis=-1)
        # x_disc_p2 = np.concatenate([batch_t, batch_i], axis=-1)
        # x_disc = np.concatenate([x_disc_p1, x_disc_p2], axis=0)
        # y_disc = np.zeros((2 * bs, 1024// pow(2, 3), 1))
        # y_disc[bs:, :, :] = 1
        #         y_disc = np.zeros(2*bs)
        #         y_disc[bs:] = 1
        # shuffle the data for discriminator
        # sh = np.arange(2 * bs)
        # np.random.shuffle(sh)
        # x_disc_sh = x_disc[sh]
        # y_disc_sh = y_disc[sh]
        # # train the discriminator once
        # d_loss = discriminator.train_on_batch(x_disc_sh, y_disc_sh)
        # # Freeze the discriminator
        # discriminator.trainable = False
        # # train the generator once
        # g_loss = generator.train_on_batch(batch_i,batch_t])
        # # Unfreeze the discriminator
        # discriminator.trainable = True
        #
        # progbar.add(bs, values=[('D logloss', disc_loss),
        #                         ('G tot', gen_loss[0]),
        #                         ('G L2', gen_loss[1]),
        #                         ('G logloss', gen_loss[2])]

    batch_v = in_dv[0:bs, :, :]
    batch_ov = out_dv[0:bs, :, :]
    fake_val = generator(batch_v, training=False)
    d_fake_val = discriminator(fake_val, training=False)
    g_loss_val2 = celoss_ones(d_fake_val)
    g_loss_val1 = K.mean(
        K.sqrt(K.sum((fake_val - batch_ov) ** 2, axis=[1, 2])) / K.sqrt(K.sum(batch_ov ** 2, axis=[1, 2])))
    g_loss_val = 1 * g_loss_val1 + 0.01*g_loss_val2



    if epoch % 1 == 0:
        print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), 'g_loss_val', float(g_loss_val))
        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))
        g_losses_val.append(float(g_loss_val))

    if epoch % 200 == 1:
        generator.save_weights('generator.ckpt')
        discriminator.save_weights('discriminator.ckpt')
        tf.saved_model.save(generator,r'D:\datahzy')
figure1=plt.figure()
x = in_dv[0:25, :, :]
x1 = out_dv[0, :, :]
x1 = x1.numpy()
fake_val1 = generator(x)
fake_val1 = fake_val1.numpy()
fake_val2 = fake_val1[0, :, :]
# fake_val2=fake_val2.reshape(256,)
y = np.arange(0, 0.02 * 1024, 0.02)
plt.plot(y, fake_val2, 'b')
plt.plot(y, x1, 'r')
figure2=plt.figure()
ep=np.arange(0,EPOCHS,1)
plt.plot(ep,g_losses_val,'r')
plt.plot(ep,g_losses,'b')
plt.show()
