from Experiments.DeepRanking.deep_rank import *
from Experiments.DeepRanking.sample_gen import *
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

if __name__ == '__main__':
    if 'weights.h5' in os.listdir('.'):
        print('Loading model.')
        model = load_model('weights.h5')
    else:
        model = deep_rank_model()
        sgd = SGD(lr=0.001)
        model.compile(sgd, loss=hinge_loss)
    model.fit_generator(train_data_generator(), 128, 128, callbacks=[ModelCheckpoint('weights.h5')])
