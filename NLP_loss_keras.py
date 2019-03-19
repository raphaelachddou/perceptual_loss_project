#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:09:42 2019

@author: antoine
"""
#%%
import pdb
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
import copy
import os

from keras.layers import Input,  ZeroPadding2D, concatenate,MaxPooling2D, Cropping2D, UpSampling2D, AveragePooling2D,Lambda,Add,Subtract
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model, load_model
from keras import backend as K

from scipy.misc import imread

import tensorflow as tf
#%%
DN_filters = []
DN_filters.append(np.array([[0,0,0,0,0],
                            [0,0,0.1011,0,0],
                            [0,0.1493,0,0.1460,0.0072],
                            [0,0,0.1015,0,0],
                            [0,0,0,0,0]]))
DN_filters.append(np.array([[0,0,0,0,0],
                            [0,0,0.0757,0,0],
                            [0,0.1986,0,0.1846,0],
                            [0,0,0.0837,0,0],
                            [0,0,0,0,0]]))
DN_filters.append(np.array([[0,0,0,0,0],
                            [0,0,0.0477,0,0],
                            [0,0.2138,0,0.2243,0],
                            [0,0,0.0467,0,0],
                            [0,0,0,0,0]]))
DN_filters.append(np.array([[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0.2503,0,0.2616,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]]))
DN_filters.append(np.array([[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0.2598,0,0.2552,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]]))
DN_filters.append(np.array([[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0.2215,0,0.0717,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]]))
DN_filters = np.array(DN_filters)

#%%
img = imread('test.jpg','L')
#%%
#img = np.expand_dims(np.expand_dims(img,axis=0),axis=-1)



def duplicate_last_row(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, -1, ...], 1)), axis=1)

def duplicate_last_col(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, :, -1, ...], 2)), axis=2)

class Laplacian_pyramid:
    def __init__(self,img_shape,N_levels):
        f = np.array([0.05,0.25, 0.4, 0.25, 0.05])
        self.filter = np.expand_dims(np.expand_dims(np.array([np.outer(f,f)]),axis=-1),axis=-1)
        self.DN_filters = np.expand_dims(np.expand_dims(DN_filters,axis=-1),axis=-1)
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.sigmas = np.array([0.0248,0.0185, 0.0179,0.0191,0.0220, 0.2782])
        self.model = self.build_model(N_levels)
        self.N_levels = N_levels


    def build_model(self,N_levels):
        #pyr = K.variable(np.array([]),dtype='float64')
        pyr = list()
        input_img = Input(shape=(self.img_rows,self.img_cols,1))
        res = input_img
        for i in range(N_levels):
            res = Conv2D(1,5,strides=(2,2),padding='same',use_bias=False)(res)
            odd1= 2*res._keras_shape[1] - self.img_rows
            odd2= 2*res._keras_shape[2] - self.img_cols
            res = Lambda(lambda t: duplicate_last_row(duplicate_last_col(t)))(res)
            res = UpSampling2D()(res)
            res = Conv2D(1,5,padding='same',use_bias=False)(res)
            res = Cropping2D(cropping=((0,odd1+2),(0,odd2+2)))(res)
            pyr.append(res)

        pyr_new = list()
        new = Subtract()([input_img,pyr[0]])
        pyr_new.append(new)
        for k in range(1,len(pyr)):
            new = Subtract()([pyr[k-1],pyr[k]])
            pyr_new.append(new)

        DN_dom = list()
        for i in range(N_levels):
            #utiliser conv2 de numpy ou bien de Keras ca c'est ok normalement
            abs_pyr = Lambda(lambda t : K.abs(t) )(pyr_new[i])
            A2 = Conv2D(1,5,padding='same',use_bias=True)(abs_pyr)
            res = Lambda(lambda inputs: tf.divide(inputs[0],inputs[1]))([pyr_new[i],A2])
            DN_dom.append(res)
        pyr_new = Lambda(lambda x: K.stack(x,axis = 1))(pyr_new)
        DN_dom= Lambda(lambda x: K.stack(x,axis = 1))(DN_dom)
        output = concatenate([pyr_new,DN_dom])
        model = Model(inputs=input_img,outputs=output)
        model.summary()
        for i in range(N_levels):
            model.layers[5*i + 1].set_weights(self.filter)
            model.layers[5*i + 4].set_weights(self.filter)
            model.layers[43+i].set_weights([self.DN_filters[i],np.array([self.sigmas[i]])])


        return model

#    def transform(self,img):
#        pyr = self.model.predict(np.expand_dims(np.expand_dims(img,axis=-1),axis=0))
#        pyr_new = [img - pyr[0][0,:,:,0]]
#        for k in range(1,len(pyr)):
#            pyr_new.append(pyr[k-1][0,:,:,0]-pyr[k][0,:,:,0])
#        return pyr_new

    def distance(self,img1,img2):
        Y_ori = self.model.predict(img1)[0,:,:,:,0]
        Lap_ori = self.model.predict(img1)[0,:,:,:,1]
        Y_dist = self.model.predict(img2)[0,:,:,:,0]
        Lap_dist = self.model.predict(img2)[0,:,:,:,1]
    #define RR_Lap_aux and RR_aux
        RR_Lap_aux = []
        RR_aux = []
        for i in range(self.N_levels):
        #utiliser les fonctions du backend keras sqrt et mean
            RR_Lap_aux.append(K.sqrt(K.mean(K.square(Lap_ori[i]-Lap_dist[i]))))
            RR_aux.append(K.sqrt(K.mean(K.square(Y_ori[i]-Y_dist[i]))))
        RR_Lap_aux = K.stack(RR_Lap_aux)
        RR_aux = K.stack(RR_aux)
        DMOS_Lap = K.mean(RR_Lap_aux)
        DMOS_Lap_dn2 = K.mean(RR_aux)

        return(DMOS_Lap)
#%%
Lap_pyr = Laplacian_pyramid(img.shape,6)
#%%
img = np.expand_dims(np.expand_dims(img,axis=0),axis=-1)
#%%
img_noisy = img + np.random.normal(scale = 2.0,size = img.shape)
#%%
pyr = Lap_pyr.model.predict(img)
a = Lap_pyr.distance(img,img_noisy)
#%%
from time import time
t1 = time()
pyr = Lap_pyr.model.predict(img)
print(time()-t1)
#%%
plt.imshow(pyr[0],'Greys')
plt.show()
plt.imshow(pyr[1],'Greys')
plt.show()
plt.imshow(pyr[2],'Greys')
plt.show()
plt.imshow(pyr[3],'Greys')
plt.show()
plt.imshow(pyr[4],'Greys')
plt.show()
plt.imshow(pyr[5],'Greys')
plt.show()
