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

import keras
from keras.layers import Input, Dense, LeakyReLU, Dropout, Lambda, subtract
from keras.models import Model
from keras import backend as K

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
        pass

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':

    def train_mac_loader(train_dataset_path, input_shape, model):

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

                if label_idx == 250: break

            label_info = {
                "label_list": label_list,
                "label_dic": label_dic,
                "label_idx": label_idx
            }

            return np.asarray(img_list), label_info

        # load train set
        img_list, label_info = train_data_loader(train_dataset_path, input_shape[:2])
        print("이미지 로드 완료 ...")

        # apply cnn filter
        num = 10
        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])

        idx = 1
        img_vecs = np.empty((len(img_list), 7, 7, 512))
        while idx * num <= len(img_list):
            img_vecs[num * (idx - 1):num * idx] = get_feature_layer([img_list[num * (idx - 1):num * idx], 0])[0]
            idx += 1

        if num * (idx - 1) != len(img_list):
            img_vecs[num * (idx - 1):] = get_feature_layer([img_list[num * (idx - 1):], 0])[0]

        del img_list
        gc.collect()
        print("맥 로드 완료 ...")

        # calculate mac feature
        img_vecs = util.cal_mac(img_vecs)

        return img_vecs, label_info

    def get_label_bound(label_info):
        label_bound = {}
        store = 0
        for i in range(label_info['label_idx']):
            label_cnt = label_info['label_list'].count(i)
            label_bound['label_' + str(i)] = [store, store + label_cnt]
            store += label_cnt

        return label_bound

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

    # load VGG16
    base_model = "vgg16"
    model = util.select_base_model(base_model)

    # bind model
    bind_model(model)

    # load weights
    nsml.load(checkpoint=base_model, session=util.model_name2session(base_model))

    # model
    model = Model(inputs=model.layers[0].input, outputs=model.get_layer('block5_pool').output)

    # parameter
    input_shape = (224, 224, 3)  # input image shape

    """ Load data """
    print('dataset path', DATASET_PATH)
    train_dataset_path = DATASET_PATH + '/train/train_data'

    img_vecs, label_info = train_mac_loader(train_dataset_path, input_shape, model)

    # l2 norm
    img_vecs = img_vecs / np.linalg.norm(img_vecs, axis=1).reshape(-1, 1)

    # similarity
    sim = util.cal_cos_sim(img_vecs, img_vecs)

    # label_bound
    label_bound = get_label_bound(label_info)

    # triplet dataset
    triplet_images = []

    for idx in range(label_info['label_idx']):

        # positive images
        label_positive = "label_" + str(idx)
        bound = label_bound[label_positive]
        vecs_positive = sim[bound[0]:bound[1], bound[0]:bound[1]]
        anchor_img, positive_img = [img_vecs[bound[0]:bound[1]][int(v[0])] for v in np.where(vecs_positive == np.amin(vecs_positive))]

        # negative images
        for idx2 in range(label_info['label_idx']):
            label_negative = "label_" + str(idx2)
            if label_positive == label_negative: continue

            bound = label_bound[label_negative]
            vecs_negative = sim[idx, bound[0]:bound[1]]
            negative_img = img_vecs[bound[0]:bound[1]][int(np.where(vecs_negative == np.amax(vecs_negative))[0][0])]

            # set triplet dataset
            triplet_images.append((anchor_img, positive_img, negative_img))

    triplet_images = np.array(triplet_images)

    anchor_images = triplet_images[:, 0]
    positive_images = triplet_images[:, 1]
    negative_images = triplet_images[:, 2]

    # triplet model

    # input
    positive_item_input = Input((512,), name='positive_item_input')
    negative_item_input = Input((512,), name='negative_item_input')
    anchor_item_input = Input((512,), name='anchor_item_input')

    # embedding
    net = build_embedding(shape=(512,), dimensions=128)
    positive_item_embedding = net(positive_item_input)
    negative_item_embedding = net(negative_item_input)
    anchor_item_embedding = net(anchor_item_input)

    # distance
    pos_dist = Lambda(lambda x: K.reshape(x, (-1, 1)))(Lambda(lambda x: K.sum(x, axis=-1))(Lambda(lambda x: K.square(x))(subtract([anchor_item_embedding, positive_item_embedding]))))
    neg_dist = Lambda(lambda x: K.reshape(x, (-1, 1)))(Lambda(lambda x: K.sum(x, axis=-1))(Lambda(lambda x: K.square(x))(subtract([anchor_item_embedding, negative_item_embedding]))))

    # loss
    basic_loss = Lambda(lambda x: x + 1)(subtract([pos_dist, neg_dist]))
    loss = Lambda(lambda x: K.maximum(x, 0.0))(basic_loss)

    # new model
    model = Model(inputs=[anchor_item_input, positive_item_input, negative_item_input], outputs=loss)
    model.summary()

    # hyper parameter
    batch_size = 64
    epochs = 100
    data_cnt = len(triplet_images)

    print(data_cnt)

    # optimizer
    opt = keras.optimizers.Adam(lr=0.0001, decay=0, amsgrad=False)
    model.compile(optimizer=opt, loss="mean_squared_error")

    hist = model.fit(x=[anchor_images, positive_images, negative_images],
                     y=[0] * data_cnt,
                     epochs=epochs,
                     batch_size=batch_size)

    print("done!")
    exit()
