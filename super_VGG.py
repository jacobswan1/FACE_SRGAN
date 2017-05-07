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
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing import image
from keras.layers.core import Dropout
from keras.engine import  Model
from keras.layers import Input, Merge
from keras_vggface import VGGFace
from scipy.misc import imsave
from Utils import conv_loss, gram_matrix, wasserstein
import numpy as np
from PIL import Image
import argparse
import math
import Data as Data


#    CNN_vgg combine
def cnn_model():    
    model = Sequential()
    model.add(Convolution2D(64, 9, 9, border_mode='same', input_shape=(3, 112, 96)))
    model.add(Activation('relu')) 
   
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(3, 3, 3, border_mode='same'))
    model.add(Activation('tanh'))
  
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        16, 5, 5,
                        border_mode='same',
                        input_shape=(3, 112, 96)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))

    model.add(Convolution2D(96, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=.001))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

#    Retrive the first image in predicting set
def combine_images(generated_images):
    generated_images = generated_images.reshape(128,112,96,3)
    image = generated_images[0,:,:,:]
    return image

def train(BATCH_SIZE):
    # Load the training data
    print('Data loading..')
    X_train, y_train, X_test, y_test = Data.loadData('data.h5')
    print('Data Loaded. Now normalizing..')

    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    y_train = (y_train.astype(np.float32) - 127.5)/127.5
    print('Data Normalized.')

    #   Reshape the img in the format of (number of rows, channels, height, weight)
    X_train = X_train.reshape((X_train.shape[0], 3) + X_train.shape[1:3])
    y_train = y_train.reshape((y_train.shape[0], 3) + y_train.shape[1:3])
    
    #    Optimization setting
    d_optim = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    g_optim = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    generator_optim = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    g_vgg_optim = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    #    Vgg model goes here
    image_input = Input(shape=(3, 112, 96))
    vgg_model = VGGFace(input_tensor=image_input, include_top=False, pooling='avg') # pooling: None, avg or max
    out = vgg_model.get_layer('pool5').output
    vgg_conv = Model(image_input, out)

    #    Generator model goes here
    generator = cnn_model()
    generator.compile(loss='mean_squared_error', optimizer= g_optim)
    
    #    Discriminative model goes here
    discriminator = discriminator_model()
    discriminator.trainable = True
    discriminator.compile(loss = 'binary_crossentropy', optimizer= d_optim)
    
    #    Gener_VGG model
    generator_vgg = \
            generator_containing_discriminator(generator, vgg_conv)
    generator_vgg.compile(
        loss=conv_loss, optimizer= g_vgg_optim)
    
    #    Gener_Discrim model
    generator_discriminator = \
            generator_containing_discriminator(generator, discriminator)
    generator_discriminator.compile(
        loss = 'binary_crossentropy', optimizer=g_optim)
        
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            lr_image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            hr_image_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(lr_image_batch, verbose=0)
            shape = generated_images.shape

            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image* 127.5+ 127.5
                imsave("./image_result/"+ str(epoch)+"_"+ str(index)+ ".png", image.astype(np.uint8))
            X = np.concatenate((hr_image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            
            #    Discriminative Model Training
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            discriminator.trainable = False
            
            #    Generator Model Training
            g_loss1 = generator.train_on_batch(lr_image_batch, hr_image_batch)
            print("batch %d gene_discri_loss : %f" % (index, g_loss1))

            
            #    Generator_Discri Model Training
            g_loss2 = generator_discriminator.train_on_batch(
               lr_image_batch, [1] * BATCH_SIZE)     
            discriminator.trainable = True
            print("batch %d gene_discri_loss : %f" % (index, g_loss2))

            
            #    Generate feature labels for the hr_images
            #labels = vgg_conv.predict(hr_image_batch)

            #g_loss2 = generator_vgg.train_on_batch(lr_image_batch, labels)
            #print("batch %d gene_vgg_loss : %f" % (index, g_loss2))
            if index % 10 == 9:
                generator.save_weights('generator', True)

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
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    elif args.mode == "mse":
        MSE(BATCH_SIZE=args.batch_size)               
