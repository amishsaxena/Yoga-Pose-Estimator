# -*- coding: utf-8 -*-
"""### Created by Amish and Jerry"""

import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from itertools import combinations
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import time
import json
import sys


def check():
    n_args = len(sys.argv)
    print("\n -----------------")
    print("Arguments recieved:", n_args, "of 3.")
    # if (n_args == 1):

    if (n_args == 3):
        print("Woring on the saved video : ", sys.argv[1])
        print("The final output will be saved at : ", sys.argv[2])
        print("\n ----------------------------------")
    else:
        print("Please run the Live Webcam model OR \n $ python3 recorded.py rec.mp4 final.avi")
        print("\n -----------------")
        sys.exit(1)

# end if wrong input
check()


# -------------------- GPU memory threshold ISSUE WORKAROUND ----------------
# 
#       UnComment the last 4 lines if an error like this pops up :
#       QObject::moveToThread: Current thread (0x7f88c0001400) is not the object's thread (0x5565f996a050).
#       Cannot move to target thread (0x7f88c0001400)
# 
# https://github.com/Leonardo-Blanger/detr_tensorflow/issues/3
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# ---------------------------------------------------------------------------


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

    # print("predicted pose --> ", a)
    # print("confidence = ", predicted_accuracy)
    # print("actual pose -->", b)
    return a, img


# json file having labels and asanas
asana_list = open("new_list.json", "rb")
inv_map = json.loads(asana_list.read())

# ---- Pretrained model for pose classification ----
# ---- For Loading the Model from internet instead of using a locally downloaded one ----
# hub_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
# movenet = model.signatures['serving_default']

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

asana_model = "./newmodel.h5"
model_fl = load_model(asana_model)

saved_video_name = sys.argv[2]
saved_video_fps = 10
rec_video = cv2.VideoCapture(sys.argv[1])

ppreds = "default"


size = (int(rec_video.get(3)), int(rec_video.get(4)))
result = cv2.VideoWriter(saved_video_name,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         saved_video_fps, size)


while True:
    ret, img = rec_video.read()
    if ret == False:
        print("Error occured while opening the video.")
        break
    
    ppreds, img_kp = predictor(img)

    img_kp = img_kp.reshape((17, 3))
    conf_kp = (img_kp[:, -1] > 0.15)
    inframe = np.all(conf_kp)

    # print("**   BODY IN FRAME  ", inframe)
    if (not inframe):
        ppreds = "full body not in frame"

    #  --- To print the detected points as the circles ---
    for body_point in range(17):
        if(img_kp[body_point, 2] > 0.25):
            cv2.circle(img, (int(img_kp[body_point, 1] * img.shape[1]), int(
                img_kp[body_point, 0] * img.shape[0])), 5, (0, 255, 0), -1)

    cv2.putText(img, str(ppreds), (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (225, 225, 0), 4)

    cv2.imshow('Frame', img)
    result.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


rec_video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The rec_video was successfully saved")


# ---------------- DUMP -----------------
# (for image manipulations during training)
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# IMG_SIZE = 256
# IMG_SIZE_LIGHT = 192
# BATCH_SIZE = 32

#  --- for FPS ---
    # cTIme=time.time()
    # fps=1/(cTIme-pTime)
    # pTime=cTIme
    # fps=round(fps,2)
    # for FPS calculation (extras)
    # pTime = 0
    
#  --- for testing outside movenet ---
    # img_kp=np.zeros(51)