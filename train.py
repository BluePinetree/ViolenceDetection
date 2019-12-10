import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import get_steps_hockey
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from data import HockeyFightDataset
from model import inceptionI3D

def main():

    # Hyperparameters
    batch = 4
    learning_rate = 0.001
    patience = 5
    weights_path = './weights'
    epochs = 50
    load_pretrained = None
    input_size = (224,224,3)

    # Load dataset
    train_generator, valid_generator = HockeyFightDataset(batch=batch, size=input_size).dataset()

    # Modeling
    inputs = Input([None, *input_size])
    predictions, end_points = inceptionI3D(inputs, dropout_keep_prob=0.5, final_endpoint='Predictions')
    i3d_model = Model(inputs, predictions)
    i3d_model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['acc'])

    # Callbacks
    callbacks = []

    tensorboard = TensorBoard()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=patience,
                                  verbose=1,
                                  mode='min',
                                  min_lr=1e-6)

    model_checkpoint = ModelCheckpoint(os.path.join(weights_path, f'I3D_{batch}batch_{epochs}epochs.h5'),
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='min')

    callbacks.append(tensorboard)
    callbacks.append(reduce_lr)
    callbacks.append(model_checkpoint)

    # Train!
    i3d_model.fit_generator(generator=train_generator,
                            steps_per_epoch=get_steps_hockey(900, batch),
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            validation_steps=get_steps_hockey(100, batch),
                            use_multiprocessing=True,
                            workers=-1)

    # Evaluate
    evaluation = i3d_model.evaluate_generator(generator=valid_generator)

    print(f'Evaluation loss : {evaluation["loss"]} , acc : {evaluation["acc"]}')




if __name__ == '__main__':
    main()
