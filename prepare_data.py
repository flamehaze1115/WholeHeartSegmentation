# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:59:36 2017

@author: Think
"""

import os
import numpy as np

import nibabel as nib

from fFindImageBoundaryCoordinate3D import fFindImageBoundaryCoordinate3D

data_dir = './Data/data_1mm'
#data_split_dir = './Data/data_1mm/Split_whole'
data_split_dir = './Data/data_1mm/WholeHeart_Test_20'

label_split_dir = os.path.join(data_split_dir,"Labels")
image_split_dir = os.path.join(data_split_dir, "Images") #分割成patch之后存储的地址

label_dir = os.path.join(data_dir, "Labels")  
image_dir = os.path.join(data_dir, "Images")  #原始的nii文件存储地址



BoundingBox_Images_dir = os.path.join(data_dir, "Images_boundingBox_Whole")
BoundingBox_Labels_dir = os.path.join(data_dir, "Labels_boundingBox_Whole")
#BoundingBox_Labels_205_dir = os.path.join(data_dir, "BoundingBox_Labels_205_smooth")
        
def changeFilename():
    #修改label的名字
    for filename in os.listdir(label_dir):
        print(filename)
        if filename.endswith("_label.nii.gz"):
            filename_new = filename.strip("_label.nii.gz") + ".nii.gz"
            print(filename_new)
            filename = os.path.join(label_dir,filename)
            filename_new = os.path.join(label_dir,filename_new)
            os.rename(filename,filename_new)

        #修改Image的名字
    for filename in os.listdir(image_dir):
        print(filename)
        if filename.endswith("_image.nii.gz"):
            filename_new = filename.strip("_image.nii.gz") + ".nii.gz"
            print(filename_new)
            filename = os.path.join(image_dir,filename)
            filename_new = os.path.join(image_dir,filename_new)
            os.rename(filename,filename_new)

def CreatNii_save(data,dir,filename,original_image):

    img = nib.Nifti1Image(data,original_image.affine)  #新建的图片与原始的affine不能变
    img.header.get_xyzt_units()
    
    nib.save(img,os.path.join(dir,filename))
  
def CutBoundingBox(img,cube_size,xdim,ydim,zdim,dir,filename):
#==============================================================================
# img: the image which should be splited
# cube_size: the size of patch_cube
# dir : the directory which the splited images store
# xdim,ydim,zdim: the coordinates of bounding box    
#==============================================================================
    xdim = xdim.astype(int)
    ydim = ydim.astype(int)
    zdim = zdim.astype(int) #convert float to int
    
    image_data = img.get_data()
    image_bouding_box = image_data[xdim[0]:xdim[1],ydim[0]:ydim[1],zdim[0]:zdim[1]]
    bounding_box_filename = filename.strip(".nii.gz")+"_boundingBox.nii.gz"
    
    CreatNii_save(image_bouding_box,dir,bounding_box_filename,img)

def FindBoundingBox():
    for label_root, dir , files in os.walk(label_dir):  #label_root=“./Data/Train\Labels”
        for file in files:  #其中一个file为“mr_train_1020_label.nii.gz”，files是file们的list
            try:
                folder = os.path.relpath(label_root, label_dir) #Return a relative filepath to path either from the current directory or from an optional start directory

#                image_root = os.path.join(image_dir, folder)
                
                label = nib.load(os.path.join(label_dir, file))
                image = nib.load(os.path.join(image_dir, file))
                
                print(label.shape)
                print(type(label.get_data()))
                xdim, ydim, zdim = fFindImageBoundaryCoordinate3D(label.get_data(),15)
                print(file)
                print("xdim: ")
                print(xdim)
                print("ydim: ")
                print(ydim)
                print("zdim: ")
                print(zdim)
                print("==================================")
                CutBoundingBox(label,30,xdim,ydim,zdim,BoundingBox_Labels_dir,file)
                CutBoundingBox(image,30,xdim,ydim,zdim,BoundingBox_Images_dir,file)
                
                
#                print ("file:"+file)
#
#                print ("label_root: "+label_root)
        
            except Exception as e:
                print(e)

def SplitImageToCubes(img,label,cube_size,stride,image_split_dir,label_split_dir,filename):
    image_data = img.get_data()
    label_data = label.get_data()
    
    number = 0
    x,y,z = label_data.shape
    
    for i in range(0,x,stride):
        for j in range(0, y, stride):
            for k in range(0, z, stride):
                if i+cube_size <= x and j+cube_size <= y and k+cube_size <= z:
                    image_cube_data = image_data[i:i+cube_size,j:j+cube_size,k:k+cube_size]
                    label_cube_data = label_data[i:i+cube_size,j:j+cube_size,k:k+cube_size]
                    
                    file_fn = filename.strip(".nii.gz") + "_" + str(number).zfill(3) + ".nii.gz"
                    number = number+1
                    
                    CreatNii_save(image_cube_data,image_split_dir,file_fn,img)
                    CreatNii_save(label_cube_data,label_split_dir,file_fn,label)
                    
#                    #方向割出几个
#    for i in reversed(range(x-2*cube_size,x,stride)):
#        for j in reversed(range(y-2*cube_size,y,stride)):
#            for k in reversed(range(z-2*cube_size,z,stride)):
#                if i-cube_size >= 0 and j-cube_size >= 0 and k-cube_size >= 0:
#                    image_cube_data = image_data[i-cube_size:i,j-cube_size:j,k-cube_size:k] 
#                    label_cube_data = label_data[i-cube_size:i,j-cube_size:j,k-cube_size:k]
#                    
#                    file_fn = filename.strip(".nii.gz") + "_" + str(number) + ".nii.gz"
#                    number = number+1
#                        
#                    CreatNii_save(image_cube_data,image_split_dir,file_fn,img)
#                    CreatNii_save(label_cube_data,label_split_dir,file_fn,label)
#                    
        
                    
    
    
    
def main():
#    FindBoundingBox()

    #os.walk(top),通过遍历目录树，自顶向下或自底向上生成目录树下的文件名。
    #对于以top 为根的目录树下的每个子目录（包括top自己），
    #它生成一个三元组(dirpath, dirnames, filenames)。
    #dirpath 为目录路径的字符串。dirnames 为dirpath 下子目录的名称列表（不包括'.' 和'..'）。
    #filenames 为dirpath 下非目录文件的名称列表。注意，列表中的名词不包含路径部分。
    #要获取dirpath 下目录或文件的完整路径（以top 开始），需要用os.path.join(dirpath, name)。
    for label_root, dir , files in os.walk(BoundingBox_Labels_dir):  #label_root=“./Data/Train\Labels”
            for file in files:  #其中一个file为“mr_train_1020_label.nii.gz”，files是file们的list
                if file.endswith('mr_train_1020_boundingBox.nii.gz'):
                    try:
                       
                        label = nib.load(os.path.join(BoundingBox_Labels_dir, file))
                        image = nib.load(os.path.join(BoundingBox_Images_dir, file))
                        
                        SplitImageToCubes(image,label,64,8,image_split_dir,label_split_dir,file)
    
                    except Exception as e:
                        print(e)


if __name__ == '__main__':
    main()