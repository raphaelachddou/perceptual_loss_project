#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:33:06 2019

@author: raphael
"""
import numpy as np


def laplacian_pyramid_s(im,N_levels):
    r,c = im.shape
    pyr = []
    f = np.array([0.05,0.25, 0.4, 0.25, 0.05])
    filtre = f.dot(f.T)
    J = np.cop(im)
    for i in range(N_levels - 1):
        res = np.convolve(J,filtre)
        res = downsample(res)
        odd = 2*res.shape - J.shape
        pyr.append(J - upsample(res,odd,filter))
        J = res
    pyr.append(J)
    return(False)
    
#%%
A = np.array([[17,25,1,8,15],[23,5,7,14,16],[4,6,13,20,22],[10,12,19,21,3],[11,18,25,2,9]])
h = np.array([-1,0,1])
np.convolve(A,h)