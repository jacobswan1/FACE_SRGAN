from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.layers.core import Dropout
from keras.engine import  Model
from keras.layers import Input
from keras_vggface import VGGFace
from scipy.misc import imsave
from Utils import conv_loss, gram_matrix
import numpy as np
from PIL import Image
import argparse
import math
import Data as Data



def cnn_model():
 
    model = Sequential()
    model.add(Flatten(input_shape=(3, 28, 24)))
    model.add(Dense(16*28*24, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(Reshape((16, 28, 24), input_shape=(16*28*24,)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    
    return model

def generator_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(3, 28, 24)))
    model.add(Activation('relu')) 
    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        16, 5, 5,
                        border_mode='same',
                        input_shape=(3, 112, 96)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(96, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

#retrive the first image in predicting set
def combine_images(generated_images):
    generated_images = generated_images.reshape(128,112,96,3)
    image = generated_images[0,:,:,:]
    return image

# Zero-center by mean pixel
def img_preprocess(img):
    img = img.astype(np.float64)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # TF order aka 'channel-last'
    x = x[:, :, :, ::-1]
    # TH order aka 'channel-first'
    # x = x[:, ::-1, :, :]
    # Zero-center by mean pixel
    x[:, 0, :, :] -= 93.5940
    x[:, 1, :, :] -= 104.7624
    x[:, 2, :, :] -= 129.1863
    return x


# Retrieve feature maps from VGG 
def retrieve_conv_feature(img_list, model):
    features = model.predict(img_list)
    return features

 # Training supervised by feature layer wise (perceptual similarity)
def train(conv_model):
    BATCH_SIZE = 128
    # load the training data
    print('Data loading..')
    X_train, y_train, X_test, y_test = Data.loadData('data_small.h5')
    print('Data Loaded. Now normalizing..')
    
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    y_train = (y_train.astype(np.float32) - 127.5)/127.5
    print('Data Normalized.')

    #   Reshape the img in the format of (number of rows, channels, height, weight)
    X_train = X_train.reshape((X_train.shape[0], 3) + X_train.shape[1:3])
    y_train = y_train.reshape((y_train.shape[0], 3) + y_train.shape[1:3])

    discriminator = discriminator_model()
    generator = cnn_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)

        
    d_optim = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    g_optim = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    generator.compile(loss= conv_loss, optimizer=g_optim)
#    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
#    discriminator.trainable = True
#    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    shape = []
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            # Random Signal Goes here (if needed)
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)

            lr_image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            hr_image_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(lr_image_batch, verbose=0)
            shape = generated_images.shape

            if index % 10 == 0:
                image = combine_images(generated_images)
                image = image* 127.5+ 127.5
                imsave("./image_result/"+str(epoch)+"_"+str(index)+".png", image.astype(np.uint8))
            X = np.concatenate((hr_image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
#            d_loss = discriminator.train_on_batch(X, y)
#            print("batch %d d_loss : %f" % (index, d_loss))

#            for i in range(BATCH_SIZE):
#                noise[i, :] = np.random.uniform(-1, 1, 100)
#            discriminator.trainable = False
            
            #retrieve conv_loss of images
            conv_lr_image_batch = retrieve_conv_feature(generated_images, conv_model)
            conv_hr_image_batch = retrieve_conv_feature(hr_image_batch, conv_model)
            
            #g_loss = 0.1* discriminator_on_generator.train_on_batch( lr_image_batch, [1] * BATCH_SIZE) 
            g_loss = generator.train_on_batch(conv_lr_image_batch, conv_hr_image_batch)
#            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
#            print(shape)
            if index % 10 == 9:
                generator.save_weights('generator', True)
#                discriminator.save_weights('discriminator', True)
   

 # Training in MSE loss function (pixel wise training)
def MSE(BATCH_SIZE):
    # load the training data
    print('Data loading..')
    X_train, y_train, X_test, y_test = Data.loadData('data_small.h5')
    print('Data Loaded. Now normalizing..')
    
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    y_train = (y_train.astype(np.float32) - 127.5)/127.5
    print('Data Normalized.')

    #   Reshape the img in the format of (number of rows, channels, height, weight)
    X_train = X_train.reshape((X_train.shape[0], 3) + X_train.shape[1:3])
    y_train = y_train.reshape((y_train.shape[0], 3) + y_train.shape[1:3])

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)

        
    d_optim = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    g_optim = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    generator.compile(loss='mean_squared_error', optimizer=g_optim)
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    shape = []
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            # Random Signal Goes here (if needed)
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)

            lr_image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            hr_image_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(lr_image_batch, verbose=0)
            shape = generated_images.shape

            if index % 10 == 0:
                image = combine_images(generated_images)
                image = image* 127.5+ 127.5
                imsave("./image_result/"+str(epoch)+"_"+str(index)+".png", image.astype(np.uint8))
            X = np.concatenate((hr_image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            

            g_loss = discriminator_on_generator.train_on_batch(
               lr_image_batch, [1] * BATCH_SIZE)
            # generator.train_on_batch(lr_image_batch, hr_image_batch)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
#            print(shape)
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)
                
                
# def generate(BATCH_SIZE, nice=False):
#     generator = generator_model()
#     generator.compile(loss='binary_crossentropy', optimizer="SGD")
#     generator.load_weights('generator')
#     if nice:
#         discriminator = discriminator_model()
#         discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
#         discriminator.load_weights('discriminator')
#         noise = np.zeros((BATCH_SIZE*20, 100))
#         for i in range(BATCH_SIZE*20):
#             noise[i, :] = np.random.uniform(-1, 1, 100)
#         generated_images = generator.predict(noise, verbose=1)
#         d_pret = discriminator.predict(generated_images, verbose=1)
#         index = np.arange(0, BATCH_SIZE*20)
#         index.resize((BATCH_SIZE*20, 1))
#         pre_with_index = list(np.append(d_pret, index, axis=1))
#         pre_with_index.sort(key=lambda x: x[0], reverse=True)
#         nice_images = np.zeros((BATCH_SIZE, 1) +
#                                (generated_images.shape[2:]), dtype=np.float32)
#         for i in range(int(BATCH_SIZE)):
#             idx = int(pre_with_index[i][1])
#             nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
#         image = combine_images(nice_images)
#     else:
#         noise = np.zeros((BATCH_SIZE, 100))
#         for i in range(BATCH_SIZE):
#             noise[i, :] = np.random.uniform(-1, 1, 100)
#         generated_images = generator.predict(noise, verbose=1)
#         image = combine_images(generated_images)
#     image = image*127.5+127.5
#     Image.fromarray(image.astype(np.uint8)).save(
# #         "./image_result/generated_image.png")




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        image_input = Input(shape=(3, 112, 96))
        vgg_model = VGGFace(input_tensor=image_input, include_top=False, pooling='avg') # pooling: None, avg or max
        out = vgg_model.get_layer('pool5').output
        vgg_conv = Model(image_input, out)
        train(vgg_conv)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    elif args.mode == "mse":
        MSE(BATCH_SIZE=args.batch_size)
