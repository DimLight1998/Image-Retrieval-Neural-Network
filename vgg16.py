import os
import random
import yaml
import numpy as np
import keras
import json
from tqdm import tqdm as tqdm
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as keras_image
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from PIL import Image


# Load configuration.
with open('config.yaml') as f:
    config = yaml.load(f)

# Load data.
if config['use_image']:
    pictures = os.listdir(config['image_path'])
    random.shuffle(pictures)
    print('Constructing features')
    features = np.zeros((len(pictures), 224, 224, 3))
    for i, image_name in tqdm(list(enumerate(pictures)), ascii=True):
        image = keras_image.load_img(os.path.join(config['image_path'], image_name), target_size=(224, 224))
        features[i, :, :, :] = keras_image.img_to_array(image)
    print('Constructing labels')
    categories = list(set(map(lambda x: x.split('_')[0], pictures)))
    category_map = {categories[i]: i for i in range(len(categories))}
    labels = np.array(list(map(lambda x: category_map[x.split('_')[0]], pictures))).reshape((-1, 1))
    labels = keras.utils.to_categorical(labels, num_classes=len(categories))


# Construct and train model.
if config['use_pretrained_vgg16']:
    model = load_model('./vgg16.h5')
elif config['use_image']:
    inputs_layer = Input(shape=(224, 224, 3))
    vgg_model = VGG16(weights='imagenet', include_top=False)
    vgg_model.trainable = False
    conv_result = vgg_model(inputs_layer)
    flattend_layer = Flatten()(conv_result)
    dense1 = Dense(512, activation='relu')(flattend_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    dense3 = Dense(10, activation='softmax')(dropout2)
    model = Model(inputs=inputs_layer, outputs=dense3)

    sgd = SGD(lr=0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    model.fit(features, labels, batch_size=4, epochs=16, validation_split=0.1)
    model.save('./vgg16.h5')
else:
    print('Error')

# Prediction.
if config['make_prediction']:
    images = os.listdir(config['test_images'])
    feature_map = {}
    for image_name in tqdm(images, ascii=True):
        image = keras_image.load_img(os.path.join(config['test_images'], image_name), target_size=(224, 224))
        feature_map[image_name] = keras_image.img_to_array(image)
    predictions = {}
    for image_name in tqdm(list(feature_map.keys()), ascii=True):
        prediction = model.predict(np.expand_dims(feature_map[image_name], 0))
        maxarg = np.argmax(prediction)
        predictions[image_name] = config['result_category_map'][maxarg]
    print(predictions)

# Update image database.
#! This method has low quality and should not be used.
if config['update_image_database']:
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    image_database_image_path = config['image_database_image_path']
    images = os.listdir(image_database_image_path)
    images_features = np.ndarray((len(images), 512))
    for i, image_name in tqdm(list(enumerate(images)), ascii=True):
        image = keras_image.load_img(os.path.join(image_database_image_path, image_name), target_size=(224, 224))
        feature = keras_image.img_to_array(image)
        feature = feature_extractor.predict(np.expand_dims(feature, 0))[0]
        images_features[i, :] = feature
    pca = PCA(n_components=64)
    pca.fit(images_features)
    pca_features = pca.transform(images_features)
    images_features = {images[i]: pca_features[i, :].tolist() for i in range(len(images))}
    images_database_keys = list(
        set(map(lambda x: x.split('_')[0], images_features.keys())))
    images_database = {key: {} for key in images_database_keys}
    for image_name in images_features.keys():
        images_database[image_name.split('_')[0]][image_name] = images_features[image_name]
    with open('database.json', 'w+') as f:
        json.dump(images_database, f)

# Nearest query in image database. (Do not use new images.)
#! This method has low quality and should not be used.
if config['find_similar_internal']:
    find_similar_internal_targets = config['find_similar_internal_targets']
    answers = {}
    with open('database.json') as f:
        database = json.load(f)
    for target_name in find_similar_internal_targets:
        category = target_name.split('_')[0]
        my_feature = database[category][target_name]
        my_neighbours = []
        for picture in database[category].keys():
            if picture == target_name:
                continue
            distance = np.linalg.norm(np.array(my_feature) - np.array(database[category][picture]))
            my_neighbours.append((picture, distance))
        my_neighbours.sort(key=lambda x: x[1])
        my_neighbours = my_neighbours[:config['find_similar_top_k']]
        answers[target_name] = my_neighbours
