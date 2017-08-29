# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:42:10 2017

@author: Think
"""
import numpy as np

#find the coordinates of bounding box of image
def fFindImageBoundaryCoordinate3D(img,offset=0):
    xdim = np.zeros(2)   #bouding box 和 x轴的交点
    ydim = np.zeros(2)   #bouding box 和 y轴的交点
    zdim = np.zeros(2)   #bouding box 和 z轴的交点
    #attention: np.zeros() the parameter should be an int or a sequence of ints
    #注意与matlab中zeros的区别，现在是想生成一个(0,0)

    tmp = np.squeeze(np.sum(np.sum(img,axis=2),axis=1))
    print("for x: ")
    print(len(tmp))
    for i in range(len(tmp)):
        if tmp[i] == 0:
            xdim[0] = i
        else:
            break
    
    xdim[1] = len(tmp)
    for i in reversed(range(len(tmp))):
        if tmp[i]==0:
            xdim[1]=i
        else:
            break
        
    #for y
    tmp = np.squeeze(np.sum(np.sum(img,axis=2),axis=0))
    print("for y: ")
    print(len(tmp))
    for i in range(len(tmp)):
        if tmp[i] == 0:
            ydim[0] = i
        else:
            break
        
    ydim[1] = len(tmp)
    for i in reversed(range(len(tmp))):
        if tmp[i]==0:
            ydim[1]=i
        else:
            break
        
    #for z
    tmp = np.squeeze(np.sum(np.sum(img,axis=1),axis=0))
    print("for z: ")
    print(len(tmp))
    for i in range(len(tmp)):
        if tmp[i] == 0:
            zdim[0] = i
        else:
            break
        
    zdim[1] = len(tmp)
    for i in reversed(range(len(tmp))):
        if tmp[i]==0:
            zdim[1]=i
        else:
            break
  
    # offset
    xdim[0] = max(0,xdim[0] - offset)
    xdim[1] = min(np.size(img,0),xdim[1] + offset)
    
    ydim[0] = max(0,ydim[0] - offset)
    ydim[1] = min(np.size(img,1),ydim[1] + offset)
    
    zdim[0] = max(0,zdim[0] - offset)
    zdim[1] = min(np.size(img,2),zdim[1] + offset)
    
    return xdim, ydim, zdim
        