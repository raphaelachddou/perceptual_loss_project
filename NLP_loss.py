#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:56:22 2019

@author: raphael
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
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from scipy.signal import convolve2d
from scipy.misc import imread
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
#class NlP_dist():
#    def __init__(self):
#        self.shape = (128,128)
#        self.n = 128
#        self.DN_filters = DN_filters
#        self.sigmas =[0.0248,0.0185, 0.0179,0.0191,0.0220, 0.2782]
#        #self.X_train = 
#        #self.Y_train =
#        
#    def laplacian_pyramid_s(im,N_levels):
#        r,c = im.shape
#        pyr = []
#        f = np.array([0.05,0.25, 0.4, 0.25, 0.05])
#        filtre = f.dot(f.T)
#        J = np.cop(im)
#        for i in range(N_levels):
#            #res = filter(J,filtre)
#            #res = downsample(res)
#            odd = 2*size(res) - size(J)
#            pyr.append(J - upsample(res,odd,filter))
#            J = res
#        pyr.append(J)
#        return(pyr)
#    
#    def NLP_transform(self,im):
#        N_levels = 6
#        #Define DN_dom with the right size
#        DN_dom = np.zeros((self.n,self.n))
#        # define laplacian_pyramid_s
#        Lap_dom = self.laplacian_pyramid_s(im,N_levels)
#        for i in range(N_levels):
#            
#            #utiliser conv2 de numpy ou bien de Keras ca c'est ok normalement
#            A2 = conv2(abs(Lap_dom[i]),self.DN_filters[i],'same')
#            
#            # division point a point 
#            DN_dom[i] = Lap_dom[i]/(self.sigma[i] + A2)
#        return(Lap_dom, DN_dom)
#    def NLP_distance(im1,im2,self):
#        N_levels = 6
#        Y_ori, Lap_ori = self.NLP_transform(im1)
#        Y_dist, Lap_dist = self.NLP_transform(im2)
#        #define RR_Lap_aux and RR_aux
#        RR_Lap_aux = np,zeros(N_levels)
#        RR_aux = np,zeros(N_levels)
#        for i in range(N_levels):
#            #utiliser les fonctions du backend keras sqrt et mean
#            RR_Lap_aux[i] = sqrt(mean((Lap_ori[i]-Lap_dist[i]).^2))
#            RR_aux[i] = sqrt(mean((Y_ori[i]-Y_dist[i]).^2))
#        DMOS_Lap = mean(RR_Lap_aux)
#        DMOS_Lap_dn2 = mean(RR_aux)
#        
#        return(DMOS_Lap,DMOS_Lap_dn2)
        
#%%

#shape = (128,128)
#n = 128
#self.X_train = 
#self.Y_train =

im  = imread('test.jpg','L')
DN_filters = DN_filters
sigmas =np.array([0.0248,0.0185, 0.0179,0.0191,0.0220, 0.2782])

    
def laplacian_pyramid_s(im,N_levels):
    r,c = im.shape
    pyr = []
    f = np.array([0.05,0.25, 0.4, 0.25, 0.05])
    f = f.reshape((1,5))
    filtre = f.T.dot(f)
    J = np.copy(im)
    for i in range(N_levels):
        res = convolve2d(J,filtre,'same','symm')
        res = res[::2,::2]
        odd1= 2*res.shape[0] - J.shape[0]
        odd2= 2*res.shape[1] - J.shape[1]
        res = np.pad(res,[1,1],'symmetric')
        res = res.repeat(2, axis=0).repeat(2, axis=1)
        res = convolve2d(res,filtre,'same','symm')
        res = res[0:r-odd1,0:c - odd2]
        pyr.append(J - res)
        J = res
    pyr.append(J)
    return(pyr)
pyr = laplacian_pyramid_s(im,6)
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
def NLP_transform(im):
    N_levels = 6
    DN_dom = []
    # define laplacian_pyramid_s
    Lap_dom = laplacian_pyramid_s(im,N_levels)
    for i in range(N_levels):
        
        #utiliser conv2 de numpy ou bien de Keras ca c'est ok normalement
        A2 = convolve2d(np.abs(Lap_dom[i]),DN_filters[i],'same','symm')
        
        # division point a point 
        DN_dom.append( np.divide(Lap_dom[i],sigmas[i] + A2 ))
    return(Lap_dom, DN_dom)

#%%
def NLP_distance(im1,im2):
    N_levels = 6
    Y_ori, Lap_ori = NLP_transform(im1)
    Y_dist, Lap_dist = NLP_transform(im2)
    #define RR_Lap_aux and RR_aux
    RR_Lap_aux = np.zeros(N_levels)
    RR_aux = np.zeros(N_levels)
    for i in range(N_levels):
        #utiliser les fonctions du backend keras sqrt et mean
        RR_Lap_aux[i] = np.sqrt(np.mean(np.square(Lap_ori[i]-Lap_dist[i])))
        RR_aux[i] = np.sqrt(np.mean(np.square(Y_ori[i]-Y_dist[i])))
    DMOS_Lap = np.mean(RR_Lap_aux)
    DMOS_Lap_dn2 = np.mean(RR_aux)
    
    return(DMOS_Lap,DMOS_Lap_dn2)
                
        
im_noisy = im + np.random.normal(scale = 20.0,size = im.shape)        
plt.imshow(im_noisy)        
plt.show()
print(NLP_distance(im,im_noisy))    
        
        
        
        
        
        
        
        
        