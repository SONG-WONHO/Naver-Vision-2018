# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import gc
import numpy as np
import cv2

import nsml
from nsml import DATASET_PATH

from keras.models import Model
from keras import backend as K

import util


def train_data_loader(train_dataset_path, img_size):
    # image list
    img_list = []

    # global variables
    label_list = []
    label_dic = {}
    label_idx = 0
    i = 0
    # train data load
    for root, dirs, files in os.walk(train_dataset_path):
        if not files: continue
        print(i)
        label_dic['label_' + str(label_idx)] = [os.path.join(root, file_name) for file_name in files if files]

        for filename in files:
            img_path = os.path.join(root, filename)

            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                i += 1
            except:
                continue

            img_list.append(img)
            label_list.append(label_idx)

        label_idx += 1

        if label_idx == 500: break

    label_info = {
        "label_list": label_list,
        "label_dic": label_dic,
        "label_idx": label_idx
    }

    return np.asarray(img_list), label_info


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):
        pass

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    # parameter
    input_shape = (224, 224, 3)  # input image shape

    """ Load data """
    print('dataset path', DATASET_PATH)
    train_dataset_path = DATASET_PATH + '/train/train_data'

    # load train set
    img_list, label_info = train_data_loader(train_dataset_path, input_shape[:2],)

    # load VGG16
    base_model = "vgg16"
    model = util.select_base_model(base_model)

    # bind model
    bind_model(model)

    # load weights
    nsml.load(checkpoint=base_model, session=util.model_name2session(base_model))

    # new model
    model = Model(inputs=model.layers[0].input, outputs=model.get_layer('block5_pool').output)

    # apply cnn filter
    num = 10
    get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])

    idx = 1
    img_vecs = np.empty((len(img_list), 7, 7, 512))
    while idx * num <= len(img_list):
        img_vecs[num * (idx - 1):num * idx] = get_feature_layer([img_list[num * (idx - 1):num * idx], 0])[0]
        idx += 1
        print(idx)

    if num * (idx - 1) != len(img_list):
        img_vecs[num * (idx - 1):] = get_feature_layer([img_list[num * (idx - 1):], 0])[0]

    del img_list, model
    gc.collect()

    # calculate mac feature
    img_vecs = util.cal_mac(img_vecs)

    # dump
    with open('./img_mac_list.pkl', 'wb') as img_f:
        pickle.dump(img_vecs, img_f)
    with open('./label_info.pkl', 'wb') as label_f:
        pickle.dump(label_info, label_f)

    print("done!")
    exit()
