import numpy as np
# from tensorflow import keras
from tensorflow import keras
import h5py
import os
import cv2

# data generator
class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_IDs,
                 labels,
                 batch_size=8,
                 dim=(288, 360, 3),
                 n_frames=41,
                 shuffle=True,
                 data_dir='./data/HockeyFights'):

        self.dim = dim
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.shuffle = shuffle
        self.data_IDs = data_IDs
        self.labels = labels
        self.data_dir = data_dir
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_IDs)) / self.batch_size)

    def __getitem__(self, idx):
        # Generate one batch of data
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # Find list of IDs
        data_IDs_temp = [self.data_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_IDs_temp):
        # X : (n_samples, *dim, n_frames)
        # Initialization
        X = np.empty((self.batch_size, self.n_frames, *self.dim), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(data_IDs_temp):
            # Store sample
            h5_file = h5py.File(os.path.join(self.data_dir, f'./processed/{ID}.h5'), 'r')
            # 전처리
            if (h5_file['data'].shape[1] > self.dim[0]) and (h5_file['data'].shape[2] > self.dim[1]):
                data = random_cropping(h5_file['data'], self.dim[:2])
            else:
                data = np.asarray([cv2.resize(im, dsize=self.dim[:2], interpolation=cv2.INTER_CUBIC)
                                   for im in h5_file['data']])

            X[i,] = data / 255.
            h5_file.close()

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=2)

# Random cropping
def random_cropping(data, crop_size=(224, 224)):
    height, width = data.shape[1], data.shape[2]
    he_idx = int(np.random.uniform(0, height - crop_size[0] + 1, size=()))
    wi_idx = int(np.random.uniform(0, width - crop_size[0] + 1, size=()))
    data = data[:, he_idx:he_idx + crop_size[0], wi_idx:wi_idx + crop_size[1]]

    return data

def get_steps_hockey(num_data, batch):
    return num_data//batch + 1