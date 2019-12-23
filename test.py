import os
import cv2
import pafy
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from data import HockeyFightDataset
from model import inceptionI3D

def main():
    input_shape = (224,224,3)
    pretrained_model = './weights/I3D_4batch_50epochs.h5'
    output_path = './samples'
    if not os.path.exists(output_path): os.mkdir(output_path)

    # data url
    url = 'https://www.youtube.com/watch?v=0dYG1tdGjQU'

    # load pretrained model
    inputs = Input([None, *input_shape])
    predictions, end_points = inceptionI3D(inputs, is_training=False, final_endpoint='Predictions', dropout_keep_prob=0.)
    i3d_model = Model(inputs, predictions)
    i3d_model.load_weights(pretrained_model)

    # get video
    video = pafy.new(url)
    print(f'title = {video.title}')
    print(f'video.rating = {video.rating}')
    print(f'video.duration = {video.duration}')

    best = video.getbest(preftype='webm')   # mp4, 3gp
    print(f'best.resolution = {best.resolution}')

    # video reader
    cap = cv2.VideoCapture(best.url)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # video writer
    forcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(os.path.join(output_path, 'test3.avi'), forcc, 30., frame_size)

    prevTime, result, vl_prob = 0, 0, 0
    input_num = 30
    input_frames = []
    while(True):
        retval, frame = cap.read()
        if retval:
            result_str = f'Violence : {result}'
            prob_str = f'Prob : {vl_prob:.4f}'

            # get frames per second
            org1 = (0, 15)
            org2 = (0, org1[1]+15)
            org3 = (0, org2[1]+15)
            str, prevTime = get_frame(prevTime)

            # put text
            put_str_img(0.3, frame, org1, str)
            if result == 0 :
                put_str_img(0.5, frame, org2, result_str)
                put_str_img(0.5, frame, org3, prob_str)
            if result == 1 :
                put_str_img(0.5, frame, org2, result_str, (0,0,255))
                put_str_img(0.5, frame, org3, prob_str, (0,0,255))

            # predict violence
            img = preprocess_image(frame, input_shape)
            input_frames.append(img)
            if len(input_frames) == input_num:
                frames = np.asarray(input_frames, dtype=np.float32)
                frames = np.expand_dims(frames, axis=0)
                result = i3d_model.predict(frames)
                # print(f'Predictions : {result[0]}')
                vl_prob = result[0][1]
                result = np.argmax(result[0])
                input_frames = []   # initialize

            out.write(frame)
            # cv2.imshow('frame', frame)

            key = cv2.waitKey(25)
            if key == 27: # esc
                break

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def preprocess_image(frame, input_shape):
    img = cv2.resize(frame, input_shape[:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img /= 255.
    return img

def get_frame(prevTime):
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = f'FPS : {fps:.2f}'
    return str, prevTime

def put_str_img(font_size, frame, org, str, color=(255,0,0)):
    size, baseLine = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
    cv2.rectangle(frame, org, (org[0] + size[0], org[1] - size[1]), color, -1)
    cv2.putText(frame, str, org, cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)


if __name__ == '__main__':
    main()