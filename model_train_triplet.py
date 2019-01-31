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
import data_loader


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

    def get_validation_idx(data_size, val_size):
        val_set_size = int(data_size * val_size)
        np.random.seed(0)

        shuffled_indices = np.random.permutation(data_size)
        train_indices = shuffled_indices[val_set_size:]
        val_indices = shuffled_indices[:val_set_size]

        return train_indices, val_indices

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

    output_path = ['mac_list_v1.pkl', 'label_info_v1.pkl']

    """ Load data """
    print('dataset path', DATASET_PATH)
    train_dataset_path = DATASET_PATH + '/train/train_data'

    nsml.cache(data_loader.train_data_mac_loader,
               data_path=train_dataset_path,
               img_size=input_shape,
               output_path=output_path,
               model=model)

    with open(output_path[0], 'rb') as img_f:
        img_vecs = pickle.load(img_f)
    with open(output_path[1], 'rb') as label_f:
        label_info = pickle.load(label_f)

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

    # validation split
    train_idx, val_idx = get_validation_idx(len(triplet_images), 0.5)

    triplet_images_val = triplet_images[val_idx]
    triplet_images = triplet_images[train_idx]

    # train anchor, positive, negative
    anchor_images = triplet_images[:, 0]
    positive_images = triplet_images[:, 1]
    negative_images = triplet_images[:, 2]

    # validation anchor, positive, negative
    anchor_images_val = triplet_images_val[:, 0]
    positive_images_val = triplet_images_val[:, 1]
    negative_images_val = triplet_images_val[:, 2]

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
    basic_loss = Lambda(lambda x: x + 10)(subtract([pos_dist, neg_dist]))
    loss = Lambda(lambda x: K.maximum(x, 0.0))(basic_loss)

    # new model
    model = Model(inputs=[anchor_item_input, positive_item_input, negative_item_input], outputs=loss)
    model.summary()

    bind_model(net)

    # hyper parameter
    batch_size = 128
    epochs = 1000
    train_size = len(triplet_images)
    val_size = len(triplet_images_val)

    print(train_size, val_size)

    # optimizer
    opt = keras.optimizers.Adam(lr=0.0001, decay=0, amsgrad=False)
    model.compile(optimizer=opt, loss="mean_squared_error")

    """ Training loop """
    for epoch in range(epochs):
        res = model.fit(x=[anchor_images, positive_images, negative_images],
                        y=[0]*train_size,
                        batch_size=batch_size,
                        initial_epoch=epoch,
                        epochs=epoch + 1,
                        verbose=1,
                        shuffle=True,
                        validation_data=([anchor_images_val, positive_images_val, negative_images_val], [0]*val_size))

        print(res.history)
        train_loss = res.history['loss'][0]
        val_loss = res.history['val_loss'][0]
        nsml.report(summary=True, step=epoch, epoch=epoch, epoch_total=epochs, loss=train_loss, val_loss=val_loss)
        if epoch % 5 == 0:
            nsml.save(epoch)

    nsml.save('saved!')

    print("done!")
    exit()
