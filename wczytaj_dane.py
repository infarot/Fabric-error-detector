import os
import cv2
import numpy as np
from listuj_sciezke import listuj_sciezke


def wczytaj_dane_train(katalog, wys, szer):
    katalogi, _ = listuj_sciezke(katalog)
    X_data = []
    Y_label = []
    for i, _ in enumerate(katalogi):
        print(katalog + '\\' + katalogi[i])
        _, pliki = listuj_sciezke(katalog + '\\' + katalogi[i])
        for j, _ in enumerate(pliki):
            plik = os.path.join(katalog + '\\' + katalogi[i], pliki[j])
            img = cv2.imread(plik)
            img = cv2.resize(img, (szer, wys), cv2.INTER_LINEAR)
            X_data.append(img)
            Y_label.append(katalogi[i])
    return [np.array(X_data, dtype=np.float64), np.array(Y_label)]


def wczytaj_dane_test(katalog, wys, szer):
    x_data = []
    y_label = []
    _, pliki = listuj_sciezke(katalog)
    for j, _ in enumerate(pliki):
        plik = os.path.join(katalog, pliki[j])
        img = cv2.imread(plik)
        img = cv2.resize(img, (szer, wys), cv2.INTER_LINEAR)
        x_data.append(img)
        y_label.append(pliki[j])
    return [np.array(x_data, dtype=np.float64), np.array(y_label)]
