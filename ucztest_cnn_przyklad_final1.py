# -*- coding: utf-8 -*-
"""
Dziala ladnie ale bez ustalania cut-offow
"""

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

# from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from categorical2label import categorical2label

from wczytaj_dane import wczytaj_dane_train, wczytaj_dane_test
from sklearn import preprocessing
# from sklearn import preprocessing

import pickle

from wylicz_ROC import wylicz_ROC

wys=32
szer=32
katalog_train="C:\CNN\DANE2\TRAIN"
katalog_test="C:\CNN\DANE2\TEST"


X_train,Y_train_label = wczytaj_dane_train(katalog_train,wys,szer)
X_test,Y_test = wczytaj_dane_test(katalog_test,wys,szer)

le = preprocessing.LabelEncoder()
le.fit(np.unique(Y_train_label))

Y_train=le.transform(Y_train_label)
liczba_klas=len(np.unique(Y_train))

tf.reset_default_graph()
X_train, Y_train = shuffle(X_train, Y_train)
Y_train = to_categorical(Y_train, liczba_klas)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, wys, szer, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, liczba_klas, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0,checkpoint_path='fabric-classifier.tfl')
model.load("fabric-classifier.tfl")
#model.fit(X_train, Y_train, n_epoch=100, shuffle=True,
#          show_metric=True, batch_size=256, run_id='fabric categorizer')

Y_test_est = model.predict(X_test)

Y_test_est_label=categorical2label(Y_test_est,textlabel=False)

Y_test_est_label = list(le.inverse_transform(Y_test_est_label))
#model.save("fabric-classifier.tfl")
#print("Network trained and saved as fabric-lassifier.tfl!")

nazwy_klas = np.unique(Y_train_label)
zgodnosc = 0
wyniki =[]
for i in range(len(Y_test)):
    if (Y_test_est_label[i][:3]==Y_test[i][:3]):
        zgodnosc += 1

    if Y_test[i][:3] == nazwy_klas[0][:3]:
        klasa = 0
    elif Y_test[i][:3] == nazwy_klas[1][:3]:
        klasa = 1
    else:
        klasa = None

    wyniki.append([Y_test_est[i][1],klasa])
    print("{} -> {} ({}) {}".format(Y_test[i],Y_test_est_label[i],Y_test_est[i], klasa))

zgodnosc = zgodnosc / len(Y_test)
print("ACC = {}".format(zgodnosc))

wylicz_ROC(wyniki,True)
#with open('wyniki.pkl', 'wb') as plik:
#     pickle.dump(wyniki, plik)

#ACC= accuracy_score(Y_test, Y_test_est2, normalize=True, sample_weight=None)
