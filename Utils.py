from keras import backend as K


# the gram matrix of an image tensor (feature-wise outer product) using shifted activations
def gram_matrix(x):
    features = K.batch_flatten(x)
    gram = K.dot(features - 1, K.transpose(features - 1))
    return gram


# vgg conv_loss
def conv_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 96 * 112
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# wasserstein loss
def wasserstein(y_true, y_pred):

    # return K.mean(y_true * y_pred) / K.mean(y_true)
    return K.mean(K.abs(y_true - y_pred))
