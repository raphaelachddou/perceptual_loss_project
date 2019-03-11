#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:56:22 2019

@author: raphael
"""
#%%
import numpy as np


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
class NlP_dist():
    def __init__(self):
        self.shape = (128,128)
        self.n = 128
        self.DN_filters = DN_filters
        self.sigmas =[0.0248,0.0185, 0.0179,0.0191,0.0220, 0.2782]
        #self.X_train = 
        #self.Y_train =
        
    def laplacian_pyramid_s(im,N_levels):
        r,c = im.shape
        pyr = []
        f = np.array([0.05,0.25, 0.4, 0.25, 0.05])
        filtre = f.dot(f.T)
        J = np.cop(im)
        for i in range(N_levels):
            #res = filter(J,filtre)
            #res = downsample(res)
            odd = 2*size(res) - size(J)
            pyr.append(J - upsample(res,odd,filter))
            J = res
        return(False)
    
    def NLP_transform(self,im):
        N_levels = 6
        #Define DN_dom with the right size
        DN_dom = np.zeros((self.n,self.n))
        # define laplacian_pyramid_s
        Lap_dom = self.laplacian_pyramid_s(im,N_levels)
        for i in range(N_levels):
            
            #utiliser conv2 de numpy ou bien de Keras ca c'est ok normalement
            A2 = conv2(abs(Lap_dom[i]),self.DN_filters[i],'same')
            
            # division point a point 
            DN_dom[i] = Lap_dom[i]/(self.sigma[i] + A2)
        return(Lap_dom, DN_dom)
    def NLP_distance(im1,im2,self):
        N_levels = 6
        Y_ori, Lap_ori = self.NLP_transform(im1)
        Y_dist, Lap_dist = self.NLP_transform(im2)
        #define RR_Lap_aux and RR_aux
        RR_Lap_aux = np,zeros(N_levels)
        RR_aux = np,zeros(N_levels)
        for i in range(N_levels):
            #utiliser les fonctions du backend keras sqrt et mean
            RR_Lap_aux[i] = sqrt(mean((Lap_ori[i]-Lap_dist[i]).^2))
            RR_aux[i] = sqrt(mean((Y_ori[i]-Y_dist[i]).^2))
        DMOS_Lap = mean(RR_Lap_aux)
        DMOS_Lap_dn2 = mean(RR_aux)
        
        return(DMOS_Lap,DMOS_Lap_dn2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        