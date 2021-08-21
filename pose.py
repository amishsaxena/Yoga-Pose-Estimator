# -*- coding: utf-8 -*-
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import pickle

import streamlit as st
# set streamlit page config
st.set_page_config(layout="wide")

from tensorflow.keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
import time
import json
from itertools import combinations
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

# -------------------- GPU memory threshold ISSUE WORKAROUND ---------------- 
# https://github.com/Leonardo-Blanger/detr_tensorflow/issues/3
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# ---------------------------------------------------------------------------


# json file having labels and asanas
asana_list = open("new_list.json", "rb")
inv_map = json.loads(asana_list.read())


def movenet_inference_flat_v10(model: tf.keras.Model, path, flip=False) -> tf.Tensor:
    """Function transforms an image to a tensor of shape [1, 17, 3]
    containing y and x coordinates and confidence level for 17 keypoints.
    :param model: movenet model in tensorflow format
    :param path: Path to image file
    :param flip: Optional argument indicating whether to flip the image left to right
    :return: Tensor with data for 17 keypoints
    """
    keypoints = model.inference_fn(get_image_live_image(path))[
        0][0].numpy().flatten()
    return keypoints


def distance(coordinates: np.array) -> tuple:
    """Function calculates distance between two keypoints
    described by x and y coordinates relative to image size.
    :param coordinates: Array with 4 values [x coordinate of the 1st keypoint,
    y coordinate of the 1st keypoint, x coordinate of the 2nd keypoint,
    y coordinate of the 2nd keypoint]
    :return: Tuple with 3 values [Euclidean distance between two points,
    distance between x coordinates, distance between y coordinates]
    """
    x_1, y_1, x_2, y_2 = coordinates
    hor_dist = abs(x_1 - x_2)
    vert_dist = abs(y_1 - y_2)
    dist = np.sqrt(hor_dist ** 2 + vert_dist ** 2)
    return dist, hor_dist, vert_dist


def is_higher(coordinates: np.array) -> int:
    """Function identifies relative positions
    of two y coordinates in vertical direction.
    :param coordinates: Array with 2 values [y coordinate of the 1st keypoint,
    y coordinate of the 2nd keypoint]
    :return: Binary value (1 - if the 1st coordinate is higher than 2nd,
    0 - if the 1st coordinate is lower than 2nd coordinate)
    """
    y_1, y_2 = coordinates
    res = int((y_1 - y_2) > 0)
    return res


def add_pos_features(df: pd.DataFrame, drop_scores=False) -> pd.DataFrame:
    """Function creates positional features based on keypoints.
    :param df: DataFrame with keypoints (x and y coordinates)
    :param drop_scores: Optional argument specifying whether to drop confidence scores
    :return: Updated DataFrame
    """
    # Distance between left and right points in pairs of limbs
    # relative to image size (Euclidean, horizontal and vertical)
    for point_type in ('elbow', 'wrist', 'knee', 'ankle'):
        d = np.apply_along_axis(
            distance, 1, df[[
                f'left_{point_type}_x', f'left_{point_type}_y',
                f'right_{point_type}_x', f'right_{point_type}_y'
            ]].values)
        df[f'{point_type}s_dist'], df[f'{point_type}s_hor_dist'], \
            df[f'{point_type}s_vert_dist'] = d.transpose()

    # Distance between specific keypoint pairs
    for point_1, point_2 in [('wrist', 'ankle'), ('wrist', 'knee'),
                             ('wrist', 'hip'), ('wrist', 'elbow'),
                             ('wrist', 'shoulder'), ('wrist', 'ear'),
                             ('ankle', 'hip'), ('ankle', 'ear'),
                             ('elbow', 'knee'), ('knee', 'hip')]:
        for side_1 in ('left', 'right'):
            for side_2 in ('left', 'right'):
                d = np.apply_along_axis(
                    distance, 1, df[[
                        f'{side_1}_{point_1}_x', f'{side_1}_{point_1}_y',
                        f'{side_2}_{point_2}_x', f'{side_2}_{point_2}_y'
                    ]].values)
                df[f'{side_1}_{point_1}_{side_2}_{point_2}_dist'], \
                    df[f'{side_1}_{point_1}_{side_2}_{point_2}_hor_dist'], \
                    df[f'{side_1}_{point_1}_{side_2}_{point_2}_vert_dist'] = d.transpose()

    # Relative upper / lower positions of specific keypoints (binary values: 0/1)
    for point_1, point_2 in combinations(['ear', 'hip', 'knee', 'ankle', 'wrist', 'elbow'], 2):
        for side_1 in ('left', 'right'):
            for side_2 in ('left', 'right'):
                df[f'{side_1}_{point_1}_{side_2}_{point_2}'] = np.apply_along_axis(
                    is_higher, 1, df[[
                        f'{side_1}_{point_1}_y', f'{side_2}_{point_2}_y'
                    ]].values)

    if drop_scores:
        columns = filter(lambda x: x.find('score') == -1, df.columns)
        df = df[columns]

    # print('Positional features added. DataFrame shape:', df.shape)

    return df


def get_image_live_image(imga):
    """Function transforms an image to a tensor.
    :param imga: the image to cast into a tensor
    :return: Tensor casted image
    """
    img = tf.image.resize_with_pad(np.expand_dims(imga, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.int32)
    return input_image


# ---- For Loading the Model from internet instead of using a locally downloaded one ----
# model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
# movenet = model.signatures['serving_default']
# Pretrained model for pose classification
# hub_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")

# define the hub_model to use
hub_model = hub.load("./movenet")
movenet = hub_model.signatures['serving_default']

# 17 keypoints in the model output
kp_descriptions = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

## (for image manipulations during training)
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# IMG_SIZE = 256
# IMG_SIZE_LIGHT = 192
# BATCH_SIZE = 32


# ---- Brain Function ----
def predictor(path):
    """Function takes the image in form of a path and outputs
    the predicted asana from the image with a numpy array of [51,1]
    from the movenet output
    :param path: pathof the image to predict the asana
    :return: predicted asana number(label) and the movenet output for it
    """
    # get keypoints from the image in a DF
    TEST_keypoints = []
    path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
    img = movenet_inference_flat_v10(hub_model, path)
    TEST_keypoints.append(img)
    TEST_keypoints_df = pd.DataFrame(TEST_keypoints)

    # Rename columns in the DataFrames according to the values
    columns = []
    for point in kp_descriptions:
        for value in ('y', 'x', 'score'):
            columns.append(f'{point}_{value}')

    TEST_keypoints_df.columns = columns
    
    # add additional positional features
    TEST_keypoints_df = add_pos_features(TEST_keypoints_df, drop_scores=True)
    # predict the asana
    prediction_existing = model_fl.predict(TEST_keypoints_df)
    # initialize the predicted_asana to 107 (no asan found)
    predicted_asana = 107

    # assign the precited asana if accuracy more than threshold (12.5%)
    for i in range(1):
        mx = 0
        mx_label = -1
        for j in range(107):
            if(prediction_existing[i, j] > mx):
                mx_label = j
                mx = prediction_existing[i, j]
        predicted_asana = mx_label
        predicted_accuracy = prediction_existing[0, mx_label]
        if(predicted_accuracy < 0.125):
            predicted_asana = 107

    # print(predicted_asana)
    
    # find label from the json
    a = inv_map[str(predicted_asana)]
    # b = "null"

    print("predicted pose --> ", a)
    print("confidence = ", predicted_accuracy)
    # print("actual pose -->", b)
    return a, img



asana_model = "./newmodel.h5"
model_fl = load_model(asana_model)


print("over")

st.title("Realtime Yoga Pose Detection")
"""### Created by Amish and Jerry"""

col1, col2 = st.beta_columns(2)
# col1, col2 = st.beta_columns(2)

FRAME_WINDOW = col1.image([])
# FRAME_WINDOW2 = col2.image([])
FRAME_WINDOW2 = col2.title(["Loading"])
FRAME_WINDOW3 = col2.subheader(["Loading"])
# FRAME_WINDOW4 = col2.subheader(["Loading"])
FRAME_WINDOW5 = col2.subheader(["Loading"])
FRAME_WINDOW6 = col2.subheader(["Loading"])
camera = cv2.VideoCapture(0)



# extras
pTime=0
pp1="default"

while True:
    _, img = camera.read()
    #  --- for FPS ---
    # cTIme=time.time()
    # fps=1/(cTIme-pTime)
    # pTime=cTIme
    # fps=round(fps,2)


    pp1, img_kp = predictor(img)
    #  --- for testing outside movenet ---
    # img_kp=np.zeros(51)
    
    img_kp = img_kp.reshape((17,3))
    conf_kp = (img_kp[:,-1] > 0.15)
    inframe = np.all(conf_kp)
    
    print("**   BODY IN FRAME  ", inframe)
    if (not inframe) : 
        pp1="full body not in frame"
        
    # Resize if image is too large
    if(img.shape[1]>720):
        img=cv2.resize(img,(    
                                int(img.shape[0]/2),
                                int(img.shape[1]/2)
                        ))

    #  --- To print the detected points as the circles ---
    # for body_point in range(17):
    #     if(img_kp[body_point,2] > 0.25):
    #         cv2.circle(img, (int(img_kp[body_point,1] * img.shape[1]), int(img_kp[body_point, 0] * img.shape[0])), 5, (0, 0, 255), -1)


    # cv2.putText(frame,str(pp1),(70,70),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255),5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(img)
    FRAME_WINDOW2.title("Your Yoga Pose")
    
    # FRAME_WINDOW2.markdown("<h2 style='text-align: center; color: black;'>"+"Your Yoga Pose"+"</h2>", unsafe_allow_html=True)
    FRAME_WINDOW2.markdown("<h2 style='text-align: center;'>"+"Your Yoga Pose"+"</h2>", unsafe_allow_html=True)

    if(pp1=="full body not in frame"):
        FRAME_WINDOW3.markdown("<h2 style='text-align: center; color: red;font-size:50px;text-transform: capitalize'>"+pp1+"</h2>", unsafe_allow_html=True)
    else:
        FRAME_WINDOW3.markdown("<h2 style='text-align: center; color: blue;font-size:50px;text-transform: capitalize'>"+pp1+"</h2>", unsafe_allow_html=True)
    # FRAME_WINDOW4.markdown("<h2 style='text-align: center;font-size:30px;text-transform: capitalize'> FPS: "+str(fps)+"</h2>", unsafe_allow_html=True)
    
    FRAME_WINDOW5.markdown("<h2 style='text-align: center;font-size:25px;text-transform: capitalize'>  Instructions for Camera Setup:</h2>", unsafe_allow_html=True)
    FRAME_WINDOW6.markdown("<h2 style='text-align: center;font-size:20px;text-transform: capitalize'> Stand more than 3 feet from the camera. <br> Make sure that your complete body is in the camera frame <br> (including toes, forehead and hands)</h2>", unsafe_allow_html=True)

else:
    st.write('Stopped')

