import numpy as np
import math

# base model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


def select_base_model(base_model):
    model = None

    # add base model what u want to download
    if base_model == "vgg16":
        model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)

    elif base_model == "vgg19":
        model = VGG19(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)

    elif base_model == "resnet50":
        model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
    else:
        print("Please Input Correct Model Name !")

    return model


def model_name2session(base_model):
    session = ""

    # add base model what u want to download
    if base_model == "vgg16":
        session = "b1ackstone/ir_ph2/4"

    elif base_model == "vgg19":
        session = "b1ackstone/ir_ph2/3"

    elif base_model == "resnet50":
        session = "b1ackstone/ir_ph2/5"

    else:
        print("Please Input Correct Model Name !")

    return session


def cal_mac(feature_vector):
    """

    :param feature_vector:
    :param featureVector: w * h * k tensor
    :return: mac value k tensor
    """

    rows = feature_vector.shape[1] * feature_vector.shape[2]
    cols = feature_vector.shape[3]

    features = np.reshape(feature_vector, (-1, rows, cols))
    features = np.amax(features, axis=1)

    return features


def cal_cos_sim(query_vec, db_vec):
    """

    :param query_vec: k tensor
    :param db_vec: k tensor
    :return: cosine similarity
    """
    cos_sim = np.dot(query_vec, db_vec.T)
    query_vec_norm = np.linalg.norm(query_vec, axis=1).reshape(-1, 1)
    db_vec_norm = np.linalg.norm(db_vec, axis=1).reshape(1, -1)

    cos_sim = cos_sim / query_vec_norm
    cos_sim = cos_sim / db_vec_norm

    return cos_sim


def cal_rmac(feature_vector, l):
    """

    :param feature_vector: [None * w * h * k] tensor (w: width, h: height, k: channel)
    :param l: layer
    :return: [None * k * m] rmac vector (k: channel, m: region)
    """
    # original image width and height, channel
    W = feature_vector.shape[1]
    H = feature_vector.shape[2]
    channel = feature_vector.shape[3]

    r_macs = []

    for layer in range(1, l + 1):
        width_region = height_region = math.ceil(2 * min(W, H) / (layer + 1))
        print(width_region, height_region)

        if layer == 1:
            x_regions = 1
            y_regions = 1

        elif layer == 2:
            x_regions = 2
            y_regions = 2

        else:
            x_regions = 3
            y_regions = 3

        coefW = W / x_regions
        coefH = H / y_regions

        for x in range(0, x_regions):
            for y in range(0, y_regions):
                initial_x = round(x * coefW)
                initial_y = round(y * coefH)
                final_x = initial_x + width_region
                final_y = initial_y + height_region

                if final_x > W:
                    final_x = W
                    initial_x = final_x - width_region

                if final_y > H:
                    final_y = H
                    initial_y = final_y - height_region

                feature_region = feature_vector[:, initial_x:final_x, initial_y:final_y, :]
                r_macs.append(cal_mac(feature_region).reshape(-1, channel, 1))

    return np.concatenate(r_macs, axis=2)


def l2_norm(feature_vector, dim):
    regions = feature_vector.shape[2]
    return np.reshape(np.linalg.norm(feature_vector, axis=dim), (-1, 1, regions))


