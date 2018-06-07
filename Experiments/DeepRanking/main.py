import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tqdm import tqdm

from Experiments.DeepRanking.deep_rank import *
from Experiments.DeepRanking.sample_gen import *
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image as keras_image
import keras.backend as K
import numpy as np

from PIL import Image


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


if __name__ == '__main__':
    mode = 'similar'
    if mode == 'train':
        if 'weights.h5' in os.listdir('.'):
            print('Loading model.')
            model = load_model('weights.h5', custom_objects={'hinge_loss': hinge_loss})
        else:
            model = deep_rank_model()
        # sgd = SGD(lr=0.001)
        sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(sgd, loss=hinge_loss)
        model.fit_generator(train_data_generator(), 128, 128, shuffle=False, callbacks=[ModelCheckpoint('weights.h5')])
    elif mode == 'index':
        print('Loading model.')
        model = load_model('weights.h5', custom_objects={'hinge_loss': hinge_loss})
        img_root = r'G:\Workspace\DS&Alg-Project1-Release\data\image'
        image_names = os.listdir(img_root)
        images = []
        features = np.zeros((len(image_names), 512))
        for i, image_name in tqdm(list(enumerate(image_names)), ascii=True):
            images.append(image_name)
            image = keras_image.load_img(os.path.join(img_root, image_name), target_size=(299, 299))
            image = keras_image.img_to_array(image)
            features[i] = model.predict(np.expand_dims(image, axis=0))
        with open('db_index.json', 'w+') as f:
            json.dump(images, f)
        np.save('db_data.npy', features)
    elif mode == 'similar':
        print('Loading model.')
        model = load_model('weights.h5', custom_objects={'hinge_loss': hinge_loss})
        img_root = r'G:\Workspace\DS&Alg-Project1-Release\data\image'
        img_name = r'n01613177_4814.JPEG'
        img = keras_image.load_img(os.path.join(img_root, img_name), target_size=(299, 299))
        img = np.expand_dims(keras_image.img_to_array(img), axis=0)
        feature = model.predict(img)
        features = np.load('db_data.npy')
        with open('db_index.json') as f:
            images = json.load(f)
        distance = [(i, np.linalg.norm(feature - features[i])) for i in range(features.shape[0])]
        distance.sort(key=lambda x: x[1])
        similar_images = list(filter(
            lambda x: x.split('_')[0] == img_name.split('_')[0], list(map(lambda x: images[x[0]], distance))))
        similar_images = similar_images[:10]

        # Check code.
        for image in similar_images:
            with open(os.path.join(img_root, image), 'rb') as f:
                im = Image.open(f)
                im.show()
                time.sleep(1)
        print(similar_images)
