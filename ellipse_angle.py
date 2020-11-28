#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:24:46 2020

@author: dev
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)

#%%
# fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))


# img = np.zeros((500, 500, 3), dtype=np.double)


# # fill ellipse
# rr, cc = ellipse(250, 250, 100, 200)#, img.shape)
# img[rr, cc, :] = 1



# ax1.imshow(img)
# #ax1.set_title('No anti-aliasing')
# ax1.axis('off')



# plt.show()
# #%%
# fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))


# img2 = np.zeros((500, 500), dtype=np.double)


# # fill ellipse
# rr, cc = ellipse(300, 300, 100, 200)#, img.shape)
# img2[rr, cc] = 1

# ax2.imshow(img2,cmap='magma')

# plt.show()
#%%
x=np.arange(np.deg2rad(0.01),np.deg2rad(180),np.deg2rad(0.05))

#x = np.random.normal(0,np.pi,500)
z = x,',',np.cos(x)

with open('label_cos.txt', 'w') as file:
    for i in range(len(x)):
        file.write('{},{}\n'.format(x[i],np.cos(x[i])))


    
    
    
    
