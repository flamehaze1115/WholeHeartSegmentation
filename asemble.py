# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:37:54 2017

@author: Think
"""


import numpy as np
import os
from scipy import ndimage
from skimage import morphology

import nibabel as nib

def CreatNii_save(data,directory,filename,affine):

    img = nib.Nifti1Image(data,affine)  #新建的图片与原始的affine不能变
    img.header.get_xyzt_units()
    
    nib.save(img,os.path.join(directory,filename))
    
#remove small islands from binary volume
def CC(Map):
    label_img, cc_num = ndimage.label(Map)
    CC = ndimage.find_objects(label_img)
    cc_areas = ndimage.sum(Map, label_img, range(cc_num+1))
    area_mask = (cc_areas < np.max(cc_areas))
    label_img[area_mask[label_img]] = 0
    return (label_img!=0).astype(np.int16)
    
#
#for file in os.listdir('./'):
#    if file.endswith(".nii.gz"):
#        label = nib.load(os.path.join('./',file))
#        data = label.get_data()
#        
#        data = (data>10).astype(np.int16)
#        filename = file.replace(".nii.gz","_1.nii.gz")
#        CreatNii_save(data,'./',filename,label.affine)

img = nib.load(os.path.join('./labels',"label_AA_whole.nii.gz"))
whole = np.zeros_like(img.get_data())

x,y,z = whole.shape

arrays=np.zeros((x,y,z,8))

for file in os.listdir('./labels/'):
    if file.endswith(".nii.gz"):
        label = nib.load(os.path.join('./labels',file))
        data = label.get_data()
        print(data.shape)
        
        if file.startswith("label_LVB_whole.nii.gz"):
            arrays[...,0]=data
            
        if file.startswith("label_RVB_whole.nii.gz"):
            arrays[...,1]=data
                    
        if file.startswith("label_LAB_whole.nii.gz"):
            arrays[...,2]=data
                    
        if file.startswith("label_RAB_whole.nii.gz"):
            arrays[...,3]=data
            
        if file.startswith("label_MLV_whole.nii.gz"):
            arrays[...,4]=data
                    
        if file.startswith("label_AA_whole.nii.gz"):
            arrays[...,5]=data
            
        if file.startswith("label_PA_whole.nii.gz"):
            arrays[...,6]=data
        if file.startswith("label_BACK_whole.nii.gz"):
            arrays[...,7]=data


incides = arrays.argmax(axis=3)
whole = np.zeros((x,y,z,8))
weights=[500,600,420,550,205,820,850,0]
for i in range(x):
    for j in range(y):
        for k in range(z):
            whole[i,j,k,incides[i,j,k]]=1 #*weights[incides[i,j,k]]
label = np.zeros((x,y,z))
whole = whole.astype(np.int16)

whole[128,...]=0
whole[129,...]=0
for i in range(184,188):
    whole[:,i,:]=0
    
for i in range(136,141):
    whole[:,:,i]=0

for i in range(8):
    Map = np.array(whole[...,i])
    print(type(Map))
    print(np.shape(Map))
    label_img = CC(Map)
    label = label +label_img *weights[i]
 

label = label.astype(np.int16)

CreatNii_save(label,'./',"whole.nii.gz",img.affine)


        
