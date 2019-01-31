# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import heapq

import nsml
import numpy as np

import keras
from keras.layers import Input, Dense, LeakyReLU, Dropout, Lambda, subtract
from keras.models import Model
from keras import backend as K

from keras.applications.vgg16 import VGG16

import util


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        # cnn filter
        m = VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000)
        get_feature_layer = K.function([m.layers[0].input] + [K.learning_phase()], [m.get_layer("block5_pool").output])

        num = 10

        # query
        idx = 1
        query_vecs = np.empty((len(query_img), 7, 7, 512))
        while idx * num <= len(query_img):
            query_vecs[num*(idx-1):num*idx] = get_feature_layer([query_img[num*(idx-1):num*idx], 0])[0]
            idx += 1

        if num*(idx-1) != len(query_img):
            query_vecs[num*(idx-1):] = get_feature_layer([query_img[num*(idx-1):], 0])[0]

        # reference
        idx = 1
        reference_vecs = np.empty((len(reference_img), 7, 7, 512))
        while idx * num <= len(reference_img):
            reference_vecs[num*(idx-1):num*idx] = get_feature_layer([reference_img[num*(idx-1):num*idx], 0])[0]
            idx += 1

        if num*(idx-1) != len(reference_img):
            reference_vecs[num*(idx-1):] = get_feature_layer([reference_img[num*(idx-1):], 0])[0]

        # shape check
        print("query vec shape: ", query_vecs.shape, " db vec shape: ", reference_vecs.shape)

        # mac values
        query_vecs = util.cal_mac(query_vecs)
        reference_vecs = util.cal_mac(reference_vecs)

        # shape check
        print("query vec shape: ", query_vecs.shape, " db vec shape: ", reference_vecs.shape)

        # l2 norm
        query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1).reshape(-1, 1)
        reference_vecs = reference_vecs / np.linalg.norm(reference_vecs, axis=1).reshape(-1, 1)

        # embedding
        # query_vecs = model.predict(query_vecs)
        # reference_vecs = model.predict(reference_vecs)

        # shape check
        print("query vec shape: ", query_vecs.shape, " db vec shape: ", reference_vecs.shape)

        # calculate cosine similarity
        sim_matrix = util.cal_cos_sim(query_vecs, reference_vecs)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = list(zip(references, sim_matrix[i].tolist()))
            sorted_sim_list = heapq.nlargest(1000, sim_list, key=lambda x: x[1])

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list[:1000]

        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


if __name__ == '__main__':

    def build_embedding(shape, dimensions):
        inp = Input(shape=shape)
        x = inp

        x = Dense(400)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.25)(x)
        x = Dense(340)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.25)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.25)(x)
        x = Dense(200)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.25)(x)

        x = Dense(dimensions)(x)
        out = x

        return Model(inputs=inp, outputs=out)

    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # embedding model

    # embedding
    model = build_embedding(shape=(512,), dimensions=128)
    model.summary()

    # bind model
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        bTrainmode = True

        # load weights
        nsml.load(checkpoint='saved!', session='b1ackstone/ir_ph2/142')
        nsml.save('saved')
        exit()
