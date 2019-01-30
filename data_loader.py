# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle
import numpy as np
import gc

from keras import backend as K

import nsml

import util


def train_data_loader(data_path, img_size, output_path):
    label_list = []
    img_list = []
    label_idx = 0
    i = 0
    for root, dirs, files in os.walk(data_path):
        print(i)
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                i+=1
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)
        label_idx += 1
        if i > 1000:
            break

    print("완료!")

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list[:35000], img_f)
    with open(output_path[1], 'wb') as img_f:
        pickle.dump(img_list[35000:], img_f)
    with open(output_path[2], 'wb') as label_f:
        pickle.dump(label_list, label_f)


def test_data_loader(data_path):
    data_path = os.path.join(data_path, 'test', 'test_data')

    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


def train_data_loader_v2(data_path, img_size, output_path):
    # image list
    img_list = []

    # global variables
    label_list = []
    label_dic = {}
    label_idx = 0
    i = 0

    # train data load
    for root, dirs, files in os.walk(data_path):
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

        if label_idx == 250: break

    label_info = {
        "label_list": label_list,
        "label_dic": label_dic,
        "label_idx": label_idx
    }

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(np.array(img_list), img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_info, label_f)


def train_data_mac_loader(data_path, img_size, output_path, model):

    # parameters
    output_path_1 = ['./image_list_280.pkl', './label_info_280.pkl']

    # load train set
    nsml.cache(train_data_loader_v2, data_path=data_path, img_size=img_size[:2], output_path=output_path_1)

    with open(output_path_1[0], 'rb') as img_f:
        img_list = pickle.load(img_f)
    with open(output_path_1[1], 'rb') as label_f:
        label_info = pickle.load(label_f)

    print("이미지 로드 완료 ...")

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

    del img_list
    gc.collect()
    print("cnn 이미지 로드 완료 ...")

    # calculate mac feature
    img_vecs = util.cal_mac(img_vecs)
    print("mac 로드 완료")

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_vecs, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_info, label_f)


if __name__ == '__main__':
    query, refer = test_data_loader('./')
    print(query)
    print(refer)
