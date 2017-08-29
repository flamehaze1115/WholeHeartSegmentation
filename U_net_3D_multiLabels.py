# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:50:03 2017

@author: Think
"""

import tensorflow as tf
import tensorlayer as tl

import numpy as np
import os

from GetData_new import GetData_new

import nibabel as nib

#change
TRAINING_DIR = './Data/Train_Split'
TEST_DIR = './Data/Test_Split'

Batch_SIZE=10
Img_rows=64
Img_cols=64
Img_depth=64
n_class = 8

RUN_NAME = "U-Net_3d"

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)   #为了存储给tensorBoard使用的数据，而新建的目录

number=0
save_dir="./Training"

smooth=1

def u_net_3d_64_1024_deconv_pro(x, n_out=8):
    """ 3-D U-Net for Image Segmentation.
    
    Parameters
    -----------
    x : tensor or placeholder of input with shape of [batch_size,depth, row, col, channel]
    batch_size : int, batch size
    n_out : int, number of output channel, default is 2 for foreground and background (binary segmentation)

    Returns
    --------
    network : TensorLayer layer class with identity output
    outputs : tensor, the output with pixel-wise softmax

    Notes
    -----
    - Recommend to use Adam with learning rate of 1e-5
    """
    tf.summary.image('input', x, max_outputs=3)
    
    batch_size = int(x._shape[0])
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    nchannel = int(x._shape[4])
    print(" * Input: size of image: %d %d %d %d" % (nx, ny, nz,nchannel))
    ## define initializer
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    ## u-net model
    # convolution
    # with tf.device('\gpu:0'):
    net_in = tl.layers.InputLayer(x, name='input')
    conv1 = tl.layers.Conv3dLayer(net_in, act=tf.nn.relu,
                shape=[3,3,3,nchannel,32], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv1') #shape of the filters, [filter_depth, filter_height, filter_width, in_channels, out_channels]
    conv1_drop = tl.layers.DropoutLayer(conv1,keep=0.5,name="drop1")
    conv2 = tl.layers.Conv3dLayer(conv1_drop, act=tf.nn.relu,
                shape=[3,3,3,32,32], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv2')
    pool1 = tl.layers.PoolLayer(conv2, ksize=[1,2,2,2,1],
                strides=[1,2,2,2,1], padding='SAME',
                pool=tf.nn.max_pool3d, name='pool1')
    
    
    
    conv3 = tl.layers.Conv3dLayer(pool1, act=tf.nn.relu,
                shape=[3,3,3,32,64], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv3')
    conv3_drop = tl.layers.DropoutLayer(conv3,keep=0.5,name="drop3")
    conv4 = tl.layers.Conv3dLayer(conv3_drop, act=tf.nn.relu,
                shape=[3,3,3,64,64], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4')
    pool2 = tl.layers.PoolLayer(conv4, ksize=[1,2,2,2,1],
                strides=[1,2,2,2,1], padding='SAME',
                pool=tf.nn.max_pool3d, name='pool2')

    
    
    conv5 = tl.layers.Conv3dLayer(pool2, act=tf.nn.relu,
                shape=[3,3,3,64,128], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv5')
    conv5_drop = tl.layers.DropoutLayer(conv5,keep=0.5,name="drop5")
    conv6 = tl.layers.Conv3dLayer(conv5_drop, act=tf.nn.relu,
                shape=[3,3,3,128,128], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv6')
    pool3 = tl.layers.PoolLayer(conv6, ksize=[1,2,2,2,1],
                strides=[1,2,2,2,1], padding='SAME',
                pool=tf.nn.max_pool3d, name='pool3')

    
    
    conv7 = tl.layers.Conv3dLayer(pool3, act=tf.nn.relu,
                shape=[3,3,3,128,256], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv7')
    conv7_drop = tl.layers.DropoutLayer(conv7,keep=0.5,name="drop7")
    conv8 = tl.layers.Conv3dLayer(conv7_drop, act=tf.nn.relu,
                shape=[3,3,3,256,256], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv8')
    # print(conv8.outputs)    # (10, 30, 30, 512)
#    pool4 = tl.layers.PoolLayer(conv8, ksize=[1,2,2,2,1],
#                strides=[1,2,2,2,1], padding='SAME',
#                pool=tf.nn.max_pool3d,name='pool4')
#    conv9 = tl.layers.Conv3dLayer(pool4, act=tf.nn.relu,
#                shape=[3,3,3,512,1024], strides=[1,1,1,1,1], padding='SAME',
#                W_init=w_init, b_init=b_init, name='conv9')
#    conv10 = tl.layers.Conv3dLayer(conv9, act=tf.nn.relu,
#                shape=[3,3,3,1024,1024], strides=[1,1,1,1,1], padding='SAME',
#                W_init=w_init, b_init=b_init, name='conv10')
#    print(" * After conv: %s" % conv10.outputs)   
#    # deconvoluation
#    deconv1 = tl.layers.DeConv3dLayer(conv10, act=tf.identity, #act=tf.nn.relu,
#                shape=[3,3,3,512,1024], strides=[1,2,2,2,1], output_shape=[batch_size,nx//8,ny//8,nz//8,512],
#                padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1')
#    # print(deconv1.outputs)  
#    deconv1_2 = tl.layers.ConcatLayer([conv8, deconv1], concat_dim=4, name='concat1_2')
#    deconv1_3 = tl.layers.Conv3dLayer(deconv1_2, act=tf.nn.relu,
#                shape=[3,3,3,1024,512], strides=[1,1,1,1,1], padding='SAME',
#                W_init=w_init, b_init=b_init, name='conv1_3')
#    deconv1_4 = tl.layers.Conv3dLayer(deconv1_3, act=tf.nn.relu,
#                shape=[3,3,3,256,256], strides=[1,1,1,1,1], padding='SAME',
#                W_init=w_init, b_init=b_init, name='conv1_4')
    deconv2 = tl.layers.DeConv3dLayer(conv8, act=tf.identity,# act=tf.nn.relu, 
                shape=[3,3,3,128,256], strides=[1,2,2,2,1], output_shape=[batch_size,nx//4,ny//4,nz//4,128],
                padding='SAME', W_init=w_init, b_init=b_init, name='devcon2_1')
    deconv2_2 = tl.layers.ConcatLayer([conv6, deconv2], concat_dim=4, name='concat2_2')
    deconv2_3 = tl.layers.Conv3dLayer(deconv2_2, act=tf.nn.relu,
                shape=[3,3,3,256,128], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv2_3')
    deconv2_3_drop = tl.layers.DropoutLayer(deconv2_3,keep=0.5,name="deconv2_3_drop")
    deconv2_4 = tl.layers.Conv3dLayer(deconv2_3_drop, act=tf.nn.relu,
                shape=[3,3,3,128,128], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv2_4')
    
    
    deconv3 = tl.layers.DeConv3dLayer(deconv2_4, act=tf.identity,# act=tf.nn.relu, 
                shape=[3,3,3,64,128], strides=[1,2,2,2,1], output_shape=[batch_size,nx//2,ny//2,nz//2,64],
                padding='SAME', W_init=w_init, b_init=b_init, name='devcon3_1')
    deconv3_2 = tl.layers.ConcatLayer([conv4, deconv3], concat_dim=4, name='concat3_2')
    deconv3_3 = tl.layers.Conv3dLayer(deconv3_2, act=tf.identity,# act=tf.nn.relu, 
                shape=[3,3,3,128,64], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv3_3')
    deconv3_3_drop = tl.layers.DropoutLayer(deconv3_3,keep=0.5,name="deconv3_3_drop")
    deconv3_4 = tl.layers.Conv3dLayer(deconv3_3_drop, act=tf.nn.relu,
                shape=[3,3,3,64,64], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv3_4')
    
    
    deconv4 = tl.layers.DeConv3dLayer(deconv3_4,act=tf.identity,# act=tf.nn.relu, 
                shape=[3,3,3,32,64], strides=[1,2,2,2,1], output_shape=[batch_size,nx,ny,nz,32],
                padding='SAME', W_init=w_init, b_init=b_init, name='devconv4_1')
    deconv4_2 = tl.layers.ConcatLayer([conv2, deconv4], concat_dim=4, name='concat4_2')
    deconv4_3 = tl.layers.Conv3dLayer(deconv4_2, act=tf.nn.relu,
                shape=[3,3,3,64,32], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4_3')
    deconv4_3_drop = tl.layers.DropoutLayer(deconv4_3,keep=0.5,name="deconv4_3_drop")
    deconv4_4 = tl.layers.Conv3dLayer(deconv4_3_drop, act=tf.nn.relu,
                shape=[3,3,3,32,32], strides=[1,1,1,1,1], padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4_4')
    

    network = tl.layers.Conv3dLayer(deconv4_4,
                act=tf.identity,
#                act=tf.sigmoid,
                shape=[1,1,1,32,n_out],       # [0]:foreground prob; [1]:background prob
                strides=[1,1,1,1,1],
                padding='SAME',
                W_init=w_init, b_init=b_init, name='conv4_5')
    # compute the softmax output
    print(" * Output: %s" % network.outputs)
    outputs = tl.act.pixel_wise_softmax(network.outputs)  #the values pf the output is between 0~1, so the labels should be 0~1
    print("outputs.softmax: %s" %outputs)
#    outputs = network.outputs

    return network, outputs


def dice_coef(y_pred,y_true):
    #dice_loss 应该由四部分组成，三个分立的部分的dice，以及心肌与血池综合起来的loss
    y_true_f = tf.reshape(y_true,[-1,n_class])
    y_pred_f = tf.reshape(y_pred,[-1,n_class])
    
    #在训练的时候发现，网络会让back所有都为1，陷入了局部最小区间，现在尝试将back的区域去除
    y_true_f = y_true_f[...,1:8]
    y_pred_f = y_pred_f[...,1:8]
    
    y_true_f = tf.cast(y_true_f,tf.float32)
    y_pred_f = tf.cast(y_pred_f,tf.float32)
    
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    union = tf.reduce_sum(tf.multiply(y_true_f , y_true_f))+ tf.reduce_sum(tf.multiply(y_pred_f , y_pred_f)) 
    return (2.*intersection+1.)/(union+1.)

def dice_coef_loss(y_pred,y_true):
    
    return 1.-dice_coef(y_pred,y_true)



def CreatNii_save(data,directory,filename,affine):

    img = nib.Nifti1Image(data,affine)  #新建的图片与原始的affine不能变
    img.header.get_xyzt_units()
    
    nib.save(img,os.path.join(directory,filename))

def main():
    #import data
    training_data = GetData_new(TRAINING_DIR)
    test_data = GetData_new(TEST_DIR)
    

    with tf.name_scope('inputs'):
        #create the model
        x=tf.placeholder(tf.float32,[Batch_SIZE,Img_depth,Img_rows,Img_cols,1],name='x_input')
        
        # Define loss and optimizer
        y_ = tf.placeholder(tf.int16, [Batch_SIZE,Img_depth, Img_rows,Img_cols,n_class],name='y__input')

    #define a global step
    global_step = tf.Variable(0,name="global_step")  

    # Build the graph for the deep net
    network, outputs= u_net_3d_64_1024_deconv_pro(x)
    
    
    dice_loss = dice_coef_loss(outputs,y_)
        
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-5).minimize(dice_loss)

    
    #add ops to save and restore all the variables
    saver = tf.train.Saver()
    
    training_summary = tf.summary.scalar("training_loss",dice_loss)
    validation_summary = tf.summary.scalar("validation_loss",dice_loss)
    
    #use only single CPU
    m_config = tf.ConfigProto()
    m_config.gpu_options.allow_growth = True

    with tf.Session(config=m_config) as sess:

        
        summary_writer = tf.summary.FileWriter("log/",sess.graph)
        
        sess.run(tf.global_variables_initializer())    #when continue training this model, should comment this line
        
        #first start to train the model, should comment these lines
#        check_points_list = tf.train.latest_checkpoint(LOG_DIR)   #return the filename of the lastest checkpoint
#        print(len(check_points_list))
#        print(check_points_list)  #is the name of this checkpoint 
#        saver.restore(sess,check_points_list)  
#        
        
        global_step_value = sess.run(global_step)
        print("Last iteration:",global_step_value)
        for i in range(global_step_value+1,150000+1):
            images,labels = training_data.next_batch(Batch_SIZE)
            feed_dict_train = {x: images, y_: labels}
            feed_dict_train.update(network.all_drop) #enable noise layers
            train_step.run(feed_dict=feed_dict_train)
            
            if i%50 == 0:
                print("iteration now:",i)
                train_loss,train_summ = sess.run([dice_loss,training_summary],feed_dict=feed_dict_train)
                summary_writer.add_summary(train_summ,i)
                print('train loss %g' % train_loss)

                
                images_test,labels_test=test_data.next_batch(Batch_SIZE)
                dp_dict = tl.utils.dict_to_one(network.all_drop)  #disable nosie layers when testing
                feed_dict_test = {x: images_test, y_: labels_test}
                feed_dict_test.update(dp_dict)
#                loss = dice_loss.eval(feed_dict=feed_dict)
                valid_loss,valid_summ = sess.run([dice_loss,validation_summary],feed_dict=feed_dict_test)
                summary_writer.add_summary(valid_summ,i)
                print('test loss %g' % valid_loss)
                print('----------------------------------')
            if i % 5000 == 0:
                print("iteration now:",i)
                
                output_image = sess.run(outputs,feed_dict=feed_dict_test)  #use the test next_batch
#                output_image = outputs.eval(feed_dict=feed_dict_test)
                print(type(output_image))
                print(np.shape(output_image))
#                output_image = np.asarray(output_image)
#                output_image= outputs.eval(feed_dict={x:images})
                for j in range(Batch_SIZE):

                    labels_test_union = labels_test[...,0]*500+labels_test[...,1]*600+labels_test[...,2]*420+labels_test[...,3]*550+labels_test[...,4]*205+labels_test[...,5]*820+labels_test[...,6]*850
                    input_Image=images_test[...,0]
                    
                    LVB = output_image[...,0]
                    out_LVB = LVB[j,...]
                    RVB = output_image[...,1]
                    out_RVB = RVB[j,...]
                    LAB = output_image[...,2]
                    out_LAB = LAB[j,...]
                    RAB = output_image[...,3]
                    out_RAB = RAB[j,...]
                    MLV = output_image[...,4]
                    out_MLV = MLV[j,...]
                    AA  = output_image[...,5]
                    out_AA = AA[j,...]
                    PA  = output_image[...,6]
                    out_PA = PA[j,...]
                    BACK = output_image[...,7]
                    out_BACK = BACK[j,...]
                    #将heart单独的label存储下来，查看效果
                    CreatNii_save(out_LVB,save_dir,"out_LVB" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_RVB,save_dir,"out_RVB" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_LAB,save_dir,"out_LAB" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_RAB,save_dir,"out_RAB" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_MLV,save_dir,"out_MLV" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_AA,save_dir,"out_AA" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_PA,save_dir,"out_PA" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save(out_BACK,save_dir,"out_BACK" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    
                    CreatNii_save(input_Image[j,...],save_dir,"Input_Test_Image" +str(i)+"_"+str(j)+ ".nii.gz",np.eye(4))
                    CreatNii_save((labels_test_union[j,...]).astype(np.float32),save_dir,"Test_Label"+str(i)+"_"+str(j) + ".nii.gz",np.eye(4)) 
              
            
            if i % 1000 == 0:
                print("iteration now:",i)
                #注意global_step.assign()并不会改变global_step的值，只是创造了这么一个操作，只有运行它之后，global_step才会真正被赋值
                global_step_op=global_step.assign(i)  #this line is necessary, if not the iteration number is always 0
                print("global_step_value:",sess.run(global_step_op))
                saver.save(sess, CHECKPOINT_FL, global_step=i)  #the "global_step" here is different from the one above
                print ("================================")
                print ("model is saved")


if __name__ == '__main__':
    main()