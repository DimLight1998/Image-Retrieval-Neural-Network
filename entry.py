import json
import os
import random
import cv2
import keras
import numpy as np
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from imgaug import augmenters as iaa
from keras import Model, Input
from keras import backend as K
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Dense, Dropout, Lambda, Flatten, Concatenate
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Conv2D, MaxPool2D
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image as keras_image
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm as tqdm

BATCH_SIZE = 3 * 3
SPLIT_NUM = 4
H_CHANNEL = 4
S_CHANNEL = 4

seq = iaa.Sequential([
    iaa.Fliplr(0.02),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.3, iaa.Superpixels(0.5, 125)),
    iaa.Add((-5, 5), True),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-10, 10),
        shear=(-5, 5)),
    iaa.OneOf([
        iaa.PerspectiveTransform(0.05),
        iaa.PiecewiseAffine(0.02)])
], random_order=True)


def get_image_hist(image_path):
    image = cv2.imread(image_path)
    num_row = image.shape[0]
    num_col = image.shape[1]
    dense_hist = np.zeros((SPLIT_NUM, SPLIT_NUM, H_CHANNEL, S_CHANNEL))
    for i in range(SPLIT_NUM):
        for j in range(SPLIT_NUM):
            img_slice = image[
                        int(i * num_row / SPLIT_NUM):int((i + 1) * num_row / SPLIT_NUM),
                        int(j * num_col / SPLIT_NUM):int((j + 1) * num_col / SPLIT_NUM)]
            hsv = cv2.cvtColor(img_slice, cv2.COLOR_BGR2HSV)
            hist_slice = cv2.calcHist([hsv], [0, 1], None, [H_CHANNEL, S_CHANNEL], [0, 180, 0, 256])
            cv2.normalize(hist_slice, hist_slice)
            while True:
                new_hist = (np.vectorize(lambda arg: 0 if arg < 0.01 else arg))(hist_slice).astype(np.float32).copy()
                cv2.normalize(new_hist, new_hist)
                if np.array_equal(new_hist, hist_slice):
                    break
                else:
                    hist_slice = new_hist
            dense_hist[i, j] = hist_slice
    return dense_hist


def get_signature_from_hist(hist):
    sig = np.array([], dtype=np.float32).reshape((0, 5))
    for i in range(SPLIT_NUM):
        for j in range(SPLIT_NUM):
            for k in range(H_CHANNEL):
                for l in range(S_CHANNEL):
                    if hist[i, j, k, l] != 0:
                        sig = np.concatenate((sig, np.array([hist[i, j, k, l], i, j, k, l]).reshape(1, 5)))
    return sig


def get_em_distance(image_path_1, image_path_2):
    hist1 = get_image_hist(image_path_1)
    sig1 = get_signature_from_hist(hist1).astype(np.float32)
    hist2 = get_image_hist(image_path_2)
    sig2 = get_signature_from_hist(hist2).astype(np.float32)
    distance = cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]
    return distance

# get_em_distance(r'/media/dz/Data/University/2018Spring/Data_Structure_and_Algorithms(2)/DS&Alg-Project1-Release/data/image/n01613177_992.JPEG',r'/media/dz/Data/University/2018Spring/Data_Structure_and_Algorithms(2)/DS&Alg-Project1-Release/data/image/n01613177_1299.JPEG')

def hinge_loss(_, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = K.variable(0, dtype='float32')
    gap = K.constant(1, dtype='float32', shape=[1])
    for i in range(0, BATCH_SIZE, 3):
        q_emb = y_pred[i + 0]
        p_emb = y_pred[i + 1]
        n_emb = y_pred[i + 2]
        dis_qp = K.sqrt(K.sum((q_emb - p_emb) ** 2))
        dis_qn = K.sqrt(K.sum((q_emb - n_emb) ** 2))
        loss += gap + dis_qp - dis_qn
    loss = loss / (BATCH_SIZE / 3)
    return K.maximum(loss, K.constant(0))


def similar_image(image: np.ndarray, n: int) -> np.ndarray:
    ret = np.zeros((n,) + image.shape)
    for i in range(n):
        ret[i] = seq.augment_image(image)
    return ret


def generate_triplet(image: str, another_image: str) -> tuple:
    query = keras_image.load_img(image, target_size=(299, 299))
    query = keras_image.img_to_array(query)
    positive = similar_image(query, 1)[0]
    negative = keras_image.load_img(another_image, target_size=(299, 299))
    negative = keras_image.img_to_array(negative)
    negative = similar_image(negative, 1)[0]
    return query, positive, negative


def train_data_generator(image_root) -> tuple:
    images = os.listdir(image_root)
    while True:
        ret_x = np.ndarray((BATCH_SIZE, 299, 299, 3))
        for i in range(int(BATCH_SIZE / 3)):
            sample = random.sample(images, 2)
            sample = list(map(lambda x: image_root + '/' + x, sample))
            triplet = generate_triplet(sample[0], sample[1])
            ret_x[3 * i + 0] = triplet[0]
            ret_x[3 * i + 1] = triplet[1]
            ret_x[3 * i + 2] = triplet[2]
        ret_y = np.zeros(BATCH_SIZE)
        yield preprocess_input(ret_x), ret_y


def sometimes(aug): return iaa.Sometimes(0.5, aug)


def deep_rank_model():
    image_input = Input(shape=(299, 299, 3))
    icpt_res = InceptionResNetV2(weights='imagenet', include_top=False)
    icpt_res.trainable = False
    conv_result = icpt_res(image_input)
    x = GlobalAveragePooling2D()(conv_result)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(x)
    convnet = Model(inputs=image_input, outputs=x)

    first_conv = AveragePooling2D(pool_size=(2, 2))(image_input)
    first_conv = Conv2D(96, kernel_size=(16, 16), strides=(8, 8), padding='same')(first_conv)
    first_max = MaxPool2D(pool_size=(5, 5), strides=(5, 5), padding='valid')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(first_max)

    second_conv = AveragePooling2D(pool_size=(2, 2))(image_input)
    second_conv = Conv2D(96, kernel_size=(16, 16), strides=(8, 8), padding='same')(second_conv)
    second_max = MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='valid')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(second_max)

    merge = Concatenate()([first_max, second_max, convnet.output])
    emb = Dense(512)(merge)
    l2_normed = Lambda(lambda arg: K.l2_normalize(arg, axis=1))(emb)
    deep_rank_model_ret = Model(inputs=image_input, outputs=l2_normed)

    return deep_rank_model_ret


def get_category_number(name, config):
    return config['category_map'][name.split('_')[0]]


if __name__ == '__main__':
    # Load configuration.
    with open('config.yaml') as f:
        config = yaml.load(f)

    mode = config['mode']

    if mode == 'train_classify':
        # Use a new model.
        input_layer = Input(shape=(299, 299, 3))
        rn_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        rn_model.trainable = False
        x = rn_model(input_layer)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=x)

        sgd = SGD(0.0002)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

        # Prepare data.
        image_root = config['image_root']
        category_map = config['category_map']
        image_names = os.listdir(image_root)
        random.shuffle(image_names)
        features = np.zeros((len(image_names), 299, 299, 3))
        for i, image_path in tqdm(list(enumerate(image_names)), ascii=True):
            image = keras_image.load_img(os.path.join(image_root, image_path), target_size=(299, 299))
            features[i] = keras_image.img_to_array(image)
        features = preprocess_input(features)
        labels = np.array(list(map(lambda arg: category_map[arg.split('_')[0]], image_names))).reshape((-1, 1))
        labels = keras.utils.to_categorical(labels, num_classes=len(category_map))

        # Start train.
        model.fit(features, labels, batch_size=4, epochs=16, validation_split=0.1)
        model.save('./resnet.h5')
    elif mode == 'classify':
        model = load_model('./resnet.h5')
        paths = config['path_files_to_be_classified']
        features = np.zeros((len(paths), 299, 299, 3))
        for i, image_path in tqdm(list(enumerate(paths)), ascii=True):
            image = keras_image.load_img(image_path, target_size=(299, 299))
            features[i] = keras_image.img_to_array(image)
        features = preprocess_input(features)
        predictions = model.predict(features)
        results = {}
        for i in range(len(paths)):
            results[paths[i]] = str(np.argmax(predictions[i]))
        with open('classify_result.json', 'w+') as f:
            json.dump(results, f)
    elif mode == 'train_rank':
        if 'deeprank.h5' in os.listdir('.'):
            model = load_model('deeprank.h5', custom_objects={'hinge_loss': hinge_loss})
        else:
            model = deep_rank_model()
            sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
            model.compile(sgd, loss=hinge_loss)
        image_root = config['image_root']
        model.fit_generator(
            train_data_generator(image_root), 128, 128, shuffle=False, callbacks=[ModelCheckpoint('deeprank.h5')])
    elif mode == 'index_rank':
        model = load_model('deeprank.h5', custom_objects={'hinge_loss': hinge_loss})
        image_root = config['image_root']
        image_names = os.listdir(image_root)
        index = []
        data = np.zeros((len(image_names), 512))
        for i, image_name in tqdm(list(enumerate(image_names)), ascii=True):
            index.append(image_name)
            image = keras_image.load_img(os.path.join(image_root, image_name), target_size=(299, 299))
            image = keras_image.img_to_array(image)
            image = preprocess_input(np.expand_dims(image, axis=0))
            data[i] = model.predict(image)[0]
        with open('db_index.json', 'w+') as f:
            json.dump(index, f)
        np.save('db_data.npy', data)
    elif mode == 'similar_rank':
        model = load_model('deeprank.h5', custom_objects={'hinge_loss': hinge_loss})
        with open('db_index.json') as f:
            index = json.load(f)
        data = np.load('db_data.npy')
        images = config['path_files_to_be_found_similar']
        image_root = config['image_root']

        result = {}
        for cat_path in images:
            image_cat = cat_path.split('@')[0]
            image_path = cat_path.split('@')[1]
            print(image_path)
            image = keras_image.load_img(image_path, target_size=(299, 299))
            image = keras_image.img_to_array(image)
            image = preprocess_input(np.expand_dims(image, axis=0))
            feature = model.predict(image)[0]
            distances = data - feature
            distance_rank = []
            for i in range(distances.shape[0]):
                if get_category_number(index[i], config) == int(image_cat):
                    distance_rank.append([np.linalg.norm(distances[i]), index[i]])
            distance_rank.sort(key=lambda arg: arg[0])
            distance_rank = distance_rank[:20]
            for i in range(len(distance_rank)):
                image_1_path = os.path.join(image_root, distance_rank[i][1])
                image_2_path = image_path
                distance_rank[i][0] = get_em_distance(image_1_path, image_2_path)
            distance_rank.sort(key=lambda arg: arg[0])
            distance_rank = distance_rank[:10]
            result[cat_path] = distance_rank
        print(result)
