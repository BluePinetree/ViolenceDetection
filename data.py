import os
import gc
import abc
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import DataGenerator, random_cropping

class VideoDataset:

    def __init__(self,
                 data_dir='./data',
                 batch=8,
                 random_seed=2019,
                 size=(224,224,3),
                 n_frames = 30):

        self.batch = batch
        self.random_seed = random_seed
        self.data_dir = data_dir
        self.size = size
        self.n_frames = n_frames

    def dataset(self):
        # 파일 경로 불러오기 후 하나의 리스트에 담아오기
        files = self._get_filepath()
        print(files[0])
        # 파일 읽어오기
        trn_ids, val_ids, total_label = self._video_dataset(files, self.data_dir)

        trn_generator = DataGenerator(data_IDs=trn_ids, labels=total_label, batch_size=self.batch, dim=self.size, shuffle=True)
        val_generator = DataGenerator(data_IDs=val_ids, labels=total_label, batch_size=self.batch, dim=self.size, shuffle=False)

        return trn_generator, val_generator

    # 파일들의 경로 리스트로 가져오기
    @abc.abstractmethod
    def _get_filepath(self):
        pass

    @abc.abstractmethod
    def _make_dataset(self, files, file_path):
        pass

    # 전처리한 파일이 있으면 받아와서 반환, 아니면 전처리
    def _video_dataset(self, files, data_path):
        file_path = self._preprocessed_dir()
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(file_path):
            if not os.path.exists(os.path.join(self.data_dir, 'processed')):
                os.mkdir(os.path.join(self.data_dir, 'processed'))
            self._make_dataset(files, file_path)

        h5_file = h5py.File(os.path.join(file_path, 'Labels.h5'), 'r')
        total_label = h5_file['labels']

        total_label = np.expand_dims(total_label, axis=-1)
        # data_ids = np.asarray(self.data_ids, dtype=int).reshape((-1,1))

        # 훈련 데이터, 검증 데이터 분리
        trn_ids, val_ids, trn_label, val_label = train_test_split(self.data_ids, total_label, test_size=0.1,
                                                                      random_state=self.random_seed)

        print(f'Train : {trn_ids.shape, trn_label.shape}\nValidation : {val_ids.shape, val_label.shape}')

        return trn_ids.ravel(), val_ids.ravel(), total_label.ravel()

    def _preprocessed_dir(self):
        return os.path.join(self.data_dir, 'processed')

class HockeyFightDataset(VideoDataset):

    def __init__(self,
                 data_dir='./data/HockeyFights',
                 batch=8,
                 random_seed=2019,
                 size=(224,224,3),
                 n_frames = 41):
        super().__init__(data_dir, batch, random_seed, size, n_frames)

    def _get_filepath(self):
        print('Configuring filepaths...')
        self.file_names = [file for file in os.listdir(self.data_dir) if file[0] != '.' and os.path.isfile(os.path.join(self.data_dir, file))]
        self.data_ids = np.arange(len(self.file_names))
        file_paths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file[0] != '.' and os.path.isfile(os.path.join(self.data_dir, file))]
        print(f'There are {len(file_paths)} files in {self.data_dir}...')
        return file_paths

    def _make_dataset(self, files, file_path):
        total_label = np.zeros(len(files), dtype=int)

        print('Preprocessing train, valid data...')
        for i, file in enumerate(tqdm(files)):
            # 프레임 단위로 저장하기
            data = np.asarray(self._get_frames(file))
            frame_df = self.n_frames - data.shape[0]
            if frame_df > 0:
                for _ in range(frame_df) : data = np.concatenate([data, np.expand_dims(data[-1], axis=0)], axis=0)
            elif frame_df < 0:
                data = data[:frame_df]

            # 저장
            h5_file = h5py.File(os.path.join(file_path, f'{i}.h5'), 'w')
            h5_file.create_dataset('data', data=data)
            h5_file.close()

            if self.file_names[i][0:2] == 'fi':
                total_label[i] = 1
            else:
                total_label[i] = 0
        print('Finished!')

        h5_file = h5py.File(os.path.join(file_path, f'Labels.h5'), 'w')
        h5_file.create_dataset('labels', data=total_label, dtype=int)
        h5_file.close()

        print(f'Successfully saved at {file_path}!')
        gc.collect()

    def _get_frames(self, file):
        cap = cv2.VideoCapture(file)
        data = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # frame = cv2.resize(frame, self.size[:2], interpolation=cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                data.append(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        data = np.asarray(data, np.uint8)
        return data



if __name__ == '__main__':
    dir = './data/HockeyFights/fi3_xvid.avi'

    # read sample
    cap = cv2.VideoCapture(dir)
    data = []
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            data.append(frame)
            print(frame.shape)
            cv2.imshow('frame', frame)
            count += 1

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break

    data = np.array(data)
    print(data.shape)

    cap.release()
    cv2.destroyAllWindows()
    print(f'Total frame number : {count}')
