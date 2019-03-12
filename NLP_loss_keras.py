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

from keras.datasets import mnist
from keras.layers import Input,  ZeroPadding2D, Concatenate,MaxPooling2D, Cropping2D, UpSampling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from scipy.signal import convolve2d
from scipy.misc import imread

import tensorflow as tf
from keras.layers import Lambda, Input
from keras.models import Model

#%%
img = imread('test.jpg','L')

def duplicate_last_row(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, -1, ...], 1)), axis=1)

def duplicate_last_col(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, :, -1, ...], 2)), axis=2)

class Laplacian_pyramid:
    def __init__(self,img_shape,N_levels):
        f = np.array([0.05,0.25, 0.4, 0.25, 0.05])
        self.filter = np.expand_dims(np.expand_dims(np.array([np.outer(f,f)]),axis=-1),axis=-1)
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.model = self.build_model(N_levels)
         
   
    def build_model(self,N_levels):
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
    
        model = Model(inputs=input_img,outputs=pyr)
        for i in range(N_levels):
            model.layers[5*i + 1].set_weights(self.filter)
            model.layers[5*i + 4].set_weights(self.filter)
        return model
    
    def transform(self,img):
        pyr = self.model.predict(np.expand_dims(np.expand_dims(img,axis=-1),axis=0))
        pyr_new = [img - pyr[0][0,:,:,0]]
        for k in range(1,len(pyr)):
            pyr_new.append(pyr[k-1][0,:,:,0]-pyr[k][0,:,:,0])
        return pyr_new
            
#%%
Lap_pyr = Laplacian_pyramid(img.shape,6)
#%%
pyr = Lap_pyr.transform(img)
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
#%%

