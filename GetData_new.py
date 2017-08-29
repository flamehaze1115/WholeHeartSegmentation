# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:19:51 2017

@author: Think
"""

import os
import random

import numpy as np

import nibabel as nib

class GetData_new():
    def __init__(self,data_dir):
        self.source_list = []

        self.label_dir = os.path.join(data_dir, "Labels")
        self.image_dir = os.path.join(data_dir, "Images")
        self.mean = 447.140490993
        self.std = 303.824365028  #这是MM-WHS 2017 whole heart segmentation 的均值和方差
        examples = 0
        
        filelist = os.listdir(self.image_dir)
        filelist.sort()
        for file in filelist:
            if not file.endswith(".nii.gz"):
                continue
            try:
                self.source_list.append(file)
                examples = examples +1
            except Exception as e:
                print(e)
        
        print("finished loading images")
        self.examples = examples
        print("Number of examples found: ", examples)
        
        
    def next_batch(self,batch_size):
        random.shuffle(self.source_list)
        images_list = []
        labels_list = []
        for i in range(batch_size):
            file = self.source_list[i]
            image = nib.load(os.path.join(self.image_dir, file))
            label = nib.load(os.path.join(self.label_dir, file))
            
            image_data = image.get_data()
            label_data = label.get_data() #image_data and label_data : shape is (x,y,z)
            
            #convert the image_data's shape to (x,y,z,channels)
            image_x,image_y,image_z = image_data.shape
            image_data = image_data[:,:,:,None]
            
            #在进行multi-label分割的时候，应该先把label变换成[size_x,size_y,size_z,n_class]的形式，把每一种label分开

            class0 = (label_data==500).astype(np.int16) #the left ventricle blood cavity
            class1 = (label_data==600).astype(np.int16) # the right ventricle blood cavity
            class2 = (label_data==420).astype(np.int16) # the left atrium blood cavity 
            class3 = (label_data==550).astype(np.int16) #the right atrium blood cavity
            class4 = (label_data==205).astype(np.int16) # the myocardium of the left ventricle
            class5 = (label_data==820).astype(np.int16) #the ascending aorta
            class6 = (label_data==850).astype(np.int16) #the pulmonary artery 
            class7 = (label_data==0).astype(np.int16) #background
            label_class = np.stack((class0,class1,class2,class3,class4,class5,class6,class7),axis=3)
            #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
            if image_data.shape==(64,64,64,1):
                image_data = (image_data-self.mean)/self.std
                images_list.append((image_data).astype(np.float32)) 
                labels_list.append((label_class).astype(np.int16))
            else:
                print("the shape of input image is not (64,64,64)")
                
        images = np.asarray(images_list)
        labels = np.asarray(labels_list)
        
        return images,labels
    
    
    
    def next_batch_order(self,batch_size):

        images_list = []
        labels_list = []
        for i in range(batch_size):
            file = self.source_list[i]
            image = nib.load(os.path.join(self.image_dir, file))
            label = nib.load(os.path.join(self.label_dir, file))
            
            image_data = image.get_data()
            label_data = label.get_data() #image_data and label_data : shape is (x,y,z)
            
            #convert the image_data's shape to (x,y,z,channels)
            image_x,image_y,image_z = image_data.shape
            image_data = image_data[:,:,:,None]
            
            #在进行multi-label分割的时候，应该先把label变换成[size_x,size_y,size_z,n_class]的形式，把每一种label分开

            class0 = (label_data==500).astype(np.int16)
            class1 = (label_data==600).astype(np.int16)
            class2 = (label_data==420).astype(np.int16)
            class3 = (label_data==550).astype(np.int16)
            class4 = (label_data==205).astype(np.int16)
            class5 = (label_data==820).astype(np.int16)
            class6 = (label_data==850).astype(np.int16)
            class7 = (label_data==0).astype(np.int16)
            label_class = np.stack((class0,class1,class2,class3,class4,class5,class6,class7),axis=3)
            #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
            if image_data.shape==(64,64,64,1):
                image_data = (image_data-self.mean)/self.std
                images_list.append((image_data).astype(np.float32)) 
                labels_list.append((label_class).astype(np.int16))
            else:
                print("the shape of input image is not (64,64,64)")
                
        images = np.asarray(images_list)
        labels = np.asarray(labels_list)
        
        return images,labels
                
    def next_batch_order_1(self,batch_size,last_point):

        images_list = []
        labels_list = []
        for i in range(last_point,last_point+batch_size):
            file = self.source_list[i]
            image = nib.load(os.path.join(self.image_dir, file))
            label = nib.load(os.path.join(self.label_dir, file))
            
            image_data = image.get_data()
            label_data = label.get_data() #image_data and label_data : shape is (x,y,z)
            
            #convert the image_data's shape to (x,y,z,channels)
            image_x,image_y,image_z = image_data.shape
            image_data = image_data[:,:,:,None]
            
            #在进行multi-label分割的时候，应该先把label变换成[size_x,size_y,size_z,n_class]的形式，把每一种label分开

            class0 = (label_data==500).astype(np.int16)
            class1 = (label_data==600).astype(np.int16)
            class2 = (label_data==420).astype(np.int16)
            class3 = (label_data==550).astype(np.int16)
            class4 = (label_data==205).astype(np.int16)
            class5 = (label_data==820).astype(np.int16)
            class6 = (label_data==850).astype(np.int16)
            class7 = (label_data==0).astype(np.int16)
            label_class = np.stack((class0,class1,class2,class3,class4,class5,class6,class7),axis=3)
            #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
            if image_data.shape==(64,64,64,1):
                image_data = (image_data-self.mean)/self.std
                images_list.append((image_data).astype(np.float32)) 
                labels_list.append((label_class).astype(np.int16))
            else:
                print("the shape of input image is not (64,64,64)")
                
        images = np.asarray(images_list)
        labels = np.asarray(labels_list)
        
        return images,labels
                
    def next_batch_order_2(self,batch_size,last_point):

        images_list = []
        for i in range(last_point,last_point+batch_size):
            file = self.source_list[i]
            image = nib.load(os.path.join(self.image_dir, file))

            
            image_data = image.get_data()
            
            #convert the image_data's shape to (x,y,z,channels)
            image_x,image_y,image_z = image_data.shape
            image_data = image_data[:,:,:,None]
            
            #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
            if image_data.shape==(64,64,64,1):
                image_data = (image_data-np.mean(image_data))/np.std(image_data)
                images_list.append((image_data).astype(np.float32)) 

            else:
                print("the shape of input image is not (64,64,64)")
                
        images = np.asarray(images_list)
        
        return images             
        
        