from Experiments.DeepRanking import similar_img_gen
from Experiments.DeepRanking.deep_rank import *
from Experiments.DeepRanking.sample_gen import *
from keras.preprocessing import image as keras_image
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

if __name__ == '__main__':
    model = deep_rank_model()
    sgd = SGD(lr=0.001)
    model.compile(sgd, loss=hinge_loss)
    model.fit_generator(train_data_genaerator, 16, epoches=128,
                        callbacks=[ModelCheckpoint(f'weights.{epoch}.h5')])
