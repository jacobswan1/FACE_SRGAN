from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.models import Sequential
from keras.layers import Reshape, merge, Input, Dense, Flatten
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import SGD, RMSprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing import image
from keras.engine import  Model
from keras_vggface import VGGFace
from scipy.misc import imsave
from Utils import conv_loss, gram_matrix, wasserstein
import numpy as np
from PIL import Image
import argparse
import math
import Data as Data


#    CNN_vgg combine, tf mode
def cnn_model():      
    
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(112, 96, 3), kernel_initializer = 'lecun_uniform'))
    model.add(Activation('relu')) 
   
    model.add(Convolution2D(64, 3, 3, border_mode='same', kernel_initializer = 'lecun_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, 3, 3, border_mode='same', kernel_initializer = 'lecun_uniform'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(3, 1, 1, border_mode='same', kernel_initializer = 'lecun_uniform'))
  
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        16, 5, 5,
                        border_mode='same',
                        input_shape=(112, 96, 3)))
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

#    Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [5, 5, stride, "same"],
                        [5, 5, (1, 1), "same"] ]
        channel_axis = 1 # for 'th' mode
        n_bottleneck_plane = n_output_plane

        #    Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=channel_axis)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis= channel_axis)(net)
                    convs = Activation("relu")(convs)
                convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                     border_mode=v[3])(convs)
            else:
                convs = BatchNormalization(axis= channel_axis)(convs)
                convs = Activation("relu")(convs)
                convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                     border_mode=v[3])(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Convolution2D(n_output_plane, nb_col=1, nb_row=1, border_mode="same")(net)
        else:
            shortcut = net

        return merge([convs, shortcut], mode="sum")
    
    return f


#    "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        return net
    return f


def res_net(input_shape):
    
    inputs = Input(shape=input_shape)

    n_stages=[128, 64, 32, 3]

    conv1 = Convolution2D(nb_filter= n_stages[0], nb_row=9, nb_col= 9, 
                          border_mode="same")(inputs) # "One conv at the beginning (spatial size: 9x 9)"

    #    Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], stride=(1,1))(conv1)# "Stage 1 (spatial size: 3x3)"
    
    conv3 = BatchNormalization(axis= 1)(conv2)
    conv3 = Activation("relu")(conv3)
    conv3 = Convolution2D(nb_filter= n_stages[2], nb_row=9, nb_col= 9, border_mode="same")(conv3) 
    
    conv4 = BatchNormalization(axis= 1)(conv3)
    conv4 = Activation("relu")(conv4)
    conv4 = Convolution2D(nb_filter= n_stages[2], nb_row=3, nb_col= 3, border_mode="same")(conv3)
    relu = Activation("relu")(conv4)
    
    conv5 = Convolution2D(nb_filter= n_stages[3], nb_row=1, nb_col= 1, border_mode="valid")(conv4) # "One conv at the beginning
    tanh = Activation("tanh")(conv5)
    
    model = Model(input=inputs, output=tanh)
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

#    Retrive the first image in predicting set
def save_image(generated_images):
    generated_images = generated_images.reshape(64, 112, 96, 3)
    image = generated_images[0,:,:,:]
    return image

def train(BATCH_SIZE):
    #    Load the training data
    print('Data loading..')
    X_train, y_train, X_test, y_test = Data.loadData('data.h5')
    print('Data Loaded. Now normalizing..')

    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    y_train = (y_train.astype(np.float32) - 127.5)/127.5
    print('Data Normalized.')
    
    #    Optimization setting RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    d_optim = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    g_vgg_optim = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    
    #    Vgg model goes here
    image_input = Input(shape=(112, 96, 3))
    vgg_model = VGGFace(input_tensor=image_input, include_top=False, pooling='avg') # pooling: None, avg or max
    out = vgg_model.get_layer('pool5').output
    vgg_conv = Model(image_input, out)

    #    Generator model goes here
    generator = res_net((112, 96, 3))
    #    Generator = cnn_model()
    generator.compile(loss='mean_squared_error', optimizer= g_optim)
    
    #    Discriminative model goes here
    discriminator = discriminator_model()
    discriminator.trainable = True
    discriminator.compile(loss = 'binary_crossentropy', optimizer= d_optim)
    
    #    Gener_VGG model
    generator_vgg = \
            generator_containing_discriminator(generator, vgg_conv)
    generator_vgg.compile(loss=conv_loss, optimizer= g_vgg_optim)
    
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

            if index % 10 == 0:
                image = save_image(generated_images)
                image = image* 127.5+ 127.5
                #imsave("./image_result/"+ str(epoch)+"_"+ str(index)+ ".png", image)
                #                imsave("./image_result/"+ str(epoch)+"_"+ str(index)+ ".png", image.astype(np.uint8))
                im = Image.fromarray(image.astype(np.uint8))
                im.save("./image_result/"+ str(epoch)+"_"+ str(index)+ ".png")

            X = np.concatenate((hr_image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            
            if epoch >= 5:
                #    Discriminative Model Training
                d_loss = discriminator.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, d_loss))
                discriminator.trainable = False

                #    Generator Model Training
                g_loss1 = generator.train_on_batch(lr_image_batch, hr_image_batch)
                print("batch %d gene_discri_loss : %f" % (index, g_loss1))

                #    Generator_Discri Model Training
                g_loss2 = generator_discriminator.train_on_batch( lr_image_batch, [1] * BATCH_SIZE)     
                discriminator.trainable = True
                print("batch %d gene_discri_loss : %f" % (index, g_loss2))
                print(' ')
            else:
                #    Discriminative Model Training
                d_loss = discriminator.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, d_loss))
                discriminator.trainable = False
                
                #    Generator Model Training
                g_loss1 = generator.train_on_batch(lr_image_batch, hr_image_batch)
                print("batch %d generator loss : %f" % (index, g_loss1))
                
                #    Generator_Discri Model Training
                g_loss2 = generator_discriminator.train_on_batch( lr_image_batch, [1] * BATCH_SIZE)     
                discriminator.trainable = True
                print("batch %d gene_discri_loss : %f" % (index, g_loss2))
            
                #    Generate feature labels for the hr_images
                labels = vgg_conv.predict(hr_image_batch)
                g_loss2 = generator_vgg.train_on_batch(lr_image_batch, labels)
                print("batch %d gene_vgg_loss : %f" % (index, g_loss2))
                
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)

def generate(BATCH_SIZE):
    #    Load the training data
    print('Data loading..')
    X_train, y_train, X_test, y_test = Data.loadData('data.h5')
    print('Data Loaded. Now normalizing..')

    X_test = (X_test.astype(np.float32) - 127.5)/127.5
    y_test = (y_test.astype(np.float32) - 127.5)/127.5
    print('Data Normalized.')
    
    #    Waiting to be completed

                
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size)
    elif args.mode == "mse":
        MSE(BATCH_SIZE=args.batch_size)               
