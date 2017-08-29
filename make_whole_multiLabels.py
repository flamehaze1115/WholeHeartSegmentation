# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:10:31 2017

@author: Think
"""

import numpy as np
import os

import nibabel as nib

def dice_coef(y_pred,y_true):
    y_true_f = np.reshape(y_true,[-1])
    y_pred_f = np.reshape(y_pred,[-1])

    y_true_f = y_true_f.astype(np.float32)
#    y_pred_f = tf.cast(y_pred_f,tf.float32)
    intersection = np.sum(np.multiply(y_true_f , y_pred_f))# tf.reduce_sum compute the sum of all the elements
    union = np.sum(np.multiply(y_true_f , y_true_f))+ np.sum(np.multiply(y_pred_f , y_pred_f)) 
    return 2. *intersection / union

def CreatNii_save(data,directory,filename,affine):

    img = nib.Nifti1Image(data,affine)  #新建的图片与原始的affine不能变
    img.header.get_xyzt_units()
    
    nib.save(img,os.path.join(directory,filename))

dir_ = './DMRI_output'

save_dir = './makewhole'

dir_label_input = './Data/DMRI/Images'  #1020.nii.gz


img_org = nib.load(os.path.join('./Data',"phase1.nii.gz"))

label_whole = np.zeros_like(img_org.get_data())
label_BACK_whole = np.zeros_like(img_org.get_data())
label_MLV_whole = np.zeros_like(img_org.get_data())
label_LVB_whole = np.zeros_like(img_org.get_data())
label_PA_whole = np.zeros_like(img_org.get_data())
label_RVB_whole = np.zeros_like(img_org.get_data())
label_LAB_whole = np.zeros_like(img_org.get_data())
label_AA_whole = np.zeros_like(img_org.get_data())
label_RAB_whole = np.zeros_like(img_org.get_data())

flag_whole  = np.zeros_like(img_org.get_data())
ones = np.ones([64,64,64])
stride = 16
cube_size = 64
batch_size = 18
number = 0
x,y,z = label_whole.shape

filelist = os.listdir(dir_)
for i in range(0,x,stride):
    for j in range(0, y, stride):
        for k in range(0, z, stride):
            if i+cube_size <= x and j+cube_size <= y and k+cube_size <= z:
                if number < 729:
                    #从乱序的输出中，找到对应的,由于未知原因，按顺序输入的patch，输出乱序
                    filename = 'Input_Test_Image116001_'+str(number)+'.nii.gz'
    
                    #由于将label分为了background，myocardium，blood pool三个部分，因此先根据存储的三个out文件，根据阈值得出最终的label   
                    label_BACK_file = filename.replace('Input_Test_Image','out_BACK')
                    label_BACK_split = nib.load(os.path.join(dir_,label_BACK_file))
                    label_BACK_split_data = label_BACK_split.get_data()
    
                    label_LVB_file = filename.replace('Input_Test_Image','out_LVB')
                    label_LVB_split = nib.load(os.path.join(dir_,label_LVB_file))
                    label_LVB_split_data = label_LVB_split.get_data()
    
                    label_RVB_file = filename.replace('Input_Test_Image','out_RVB')
                    label_RVB_split = nib.load(os.path.join(dir_,label_RVB_file))
                    label_RVB_split_data = label_RVB_split.get_data()
    
                    label_LAB_file = filename.replace('Input_Test_Image','out_LAB')
                    label_LAB_split = nib.load(os.path.join(dir_,label_LAB_file))
                    label_LAB_split_data = label_LAB_split.get_data()
                    
                    label_RAB_file = filename.replace('Input_Test_Image','out_RAB')
                    label_RAB_split = nib.load(os.path.join(dir_,label_RAB_file))
                    label_RAB_split_data = label_RAB_split.get_data()
                    
                    label_MLV_file = filename.replace('Input_Test_Image','out_MLV')
                    label_MLV_split = nib.load(os.path.join(dir_,label_MLV_file))
                    label_MLV_split_data = label_MLV_split.get_data()
    
                    label_AA_file = filename.replace('Input_Test_Image','out_AA')
                    label_AA_split = nib.load(os.path.join(dir_,label_AA_file))
                    label_AA_split_data = label_AA_split.get_data()
    
                    label_PA_file = filename.replace('Input_Test_Image','out_PA')
                    label_PA_split = nib.load(os.path.join(dir_,label_PA_file))
                    label_PA_split_data = label_PA_split.get_data()                
                    
    #                arrays=[label_BACK_split_data,label_LVB_split_data,label_RVB_split_data,label_LAB_split_data,label_RAB_split_data,label_MLV_split_data,label_AA_split_data,label_PA_split_data]
    #                cube=np.stack(arrays, axis=3)
    #                indices = np.argmax(cube,3)
    #                b = np.zeros_like(cube)
    #                b[indices]=1
    
                    
                    setoff = 0.6
                    label_BACK_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_BACK_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_BACK_split_data>setoff).astype(np.int16)
                                  
                    label_LVB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_LVB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_LVB_split_data>setoff).astype(np.int16)  
                    label_RVB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_RVB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_RVB_split_data>setoff).astype(np.int16)  
                    label_LAB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_LAB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_LAB_split_data>setoff).astype(np.int16)  
                    label_RAB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_RAB_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_RAB_split_data>setoff).astype(np.int16)  
                    label_MLV_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_MLV_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_MLV_split_data>setoff).astype(np.int16)  
                    label_AA_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_AA_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_AA_split_data>setoff).astype(np.int16)  
                    label_PA_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = label_PA_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+(label_PA_split_data>setoff).astype(np.int16)  
    
                    
                    flag_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size] = flag_whole[i:i+cube_size,j:j+cube_size,k:k+cube_size]+ones
                number = number+1

                

flag_whole = flag_whole.astype(np.float32)

label_BACK_whole = label_BACK_whole.astype(np.float32)
label_MLV_whole = label_MLV_whole.astype(np.float32)
label_LVB_whole = label_LVB_whole.astype(np.float32)
label_PA_whole = label_PA_whole.astype(np.float32)
label_RVB_whole = label_RVB_whole.astype(np.float32)
label_LAB_whole = label_LAB_whole.astype(np.float32)
label_AA_whole = label_AA_whole.astype(np.float32)
label_RAB_whole = label_RAB_whole.astype(np.float32)

x,y,z = flag_whole.shape


wholes=[label_BACK_whole,label_MLV_whole,label_LVB_whole,label_PA_whole,label_RVB_whole,label_LAB_whole,label_AA_whole,label_RAB_whole]
filenames=["label_BACK_whole.nii.gz","label_MLV_whole.nii.gz","label_LVB_whole.nii.gz","label_PA_whole.nii.gz","label_RVB_whole.nii.gz","label_LAB_whole.nii.gz","label_AA_whole.nii.gz","label_RAB_whole.nii.gz",]

setoff = 0.01

for i in range(8):
    temp = (wholes[i]).astype(np.float32)
    CreatNii_save(temp,save_dir,filenames[i],img_org.affine)



