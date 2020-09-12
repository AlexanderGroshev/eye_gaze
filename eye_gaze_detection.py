import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import pylab
import os
from keras.preprocessing import image
import time

#Функции
def get_left_eye(eyes): #Детекция левого глаза
    l=9000
    index_l=-1
    for i in range(len(eyes)):
        if eyes[i][0]<l:
            l=eyes[i][0]
            index_l=i
    return eyes[index_l]

def resize_image(input_image_path, size): #Изменение размерности изображения
    original_image = input_image_path
    width, height = original_image.size
    resized_image = original_image.resize(size)
    width, height = resized_image.size
    return resized_image

def cut_eye(img): #Обрезание нужной области
    height, width = img.shape[:2]
    cut_area = int(height / 5)
    img = img[cut_area:height-cut_area, 0:width]
    return img

def predict_(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (500, 650))
    try:
        eyes = eye_cascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 5)
        left = get_left_eye(eyes).reshape(1, 4)
        ex,ey,ew,eh = left[0]
        eye_image = image [ey:ey+eh, ex:ex+ew]
        gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        gray_eye = cut_eye(gray_eye)
        bigger = cv2.resize(gray_eye, (50, 76))
        bigger = np.array(bigger)
        lst = (model.predict(np.array([bigger.reshape(76, 50, 1)]))[0])
        index = np.argmax(lst)
        if (index == 0):
            return 1
        else:
            return 0
    except:
        return 0


#Модель нейронной сети
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(76, 50, 1), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation ='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation='softmax'))
#Загрузка предобученных весов
model.load_weights('model_weights.h5')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
index_f = 0 #Счетчик кадров

print ('Выберите иточник данных: \n1 - изображение из файла\n2 - изображение с веб-камеры\n3 - видео из файла')
date = int(input())
if (date == 1):
    path_image = input('Введите путь до изображения: ')
    print(predict_(path_image))

if (date == 2) or (date == 3):
    if (date == 2):
        cap = cv2.VideoCapture(0)
    else:
        path_video = input('Введите путь до видеофайла: ')
        cap = cv2.VideoCapture(path_video)

    while True:
        #Получение картинки
        _, img = cap.read()

        #Оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Нахождение лица
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        try:
            left = get_left_eye(eyes).reshape(1, 4)
            ex, ey, ew, eh = left[0]
            eye_image = gray[ey:ey + eh, ex:ex + ew]
            bigger = cv2.resize(cut_eye(eye_image), (50, 76))
            bigger = np.array(bigger)
            lst = (model.predict(np.array([bigger.reshape(76, 50, 1)]))[0])
            index = np.argmax(lst)

            if (index == 0):
                print('#', index_f, 'LOOKING AT CAM')
                index_f +=1
            else:
                print ('#',index_f, 'NOT LOOKING AT CAM')
                index_f += 1

        except:
            print('#',index_f, 'NO EYE FOUND')
            index_f += 1

        cv2.imshow('img', img)

        #Стоп
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()