from keras.applications import VGG16
from keras.layers import Dense, Dropout, Lambda, Flatten, Concatenate
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Conv2D, MaxPool2D
from keras import Model, Input
from keras import backend as K

BATCH_SIZE = 1 * 3


def deep_rank_model():
    image_input = Input(shape=(225, 225, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False)
    vgg16.trainable = False
    conv_result = vgg16(image_input)
    x = GlobalAveragePooling2D()(conv_result)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(x)
    convnet = Model(inputs=image_input, outputs=x)

    first_conv = AveragePooling2D(pool_size=(2, 2))(image_input)
    first_conv = Conv2D(96, kernel_size=(8, 8), strides=(4, 4), padding='same')(first_conv)
    first_max = MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='valid')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(first_max)

    second_conv = AveragePooling2D(pool_size=(2, 2))(image_input)
    second_conv = Conv2D(96, kernel_size=(8, 8), strides=(4, 4), padding='same')(second_conv)
    second_max = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(second_max)

    merge = Concatenate()([first_max, second_max, convnet.output])
    emb = Dense(1024)(merge)
    l2_normed = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(emb)
    deep_rank_model_ret = Model(inputs=image_input, outputs=l2_normed)

    return deep_rank_model_ret



