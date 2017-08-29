clear all;clc;close all;

%% rotate 3D MRI 
% file_image_dir='E:/UCLA_summer_intern/3D_segmentation/U_Net_3D/Data/Images/';
% file_label_dir='E:/UCLA_summer_intern/3D_segmentation/U_Net_3D/Data/Labels/';

file_image_dir='E:/UCLA_summer_intern/3D_segmentation/U_Net_3D/Data/data_1mm/Images/';
file_label_dir='E:/UCLA_summer_intern/3D_segmentation/U_Net_3D/Data/data_1mm/Labels/';

subjImageName = dir([file_image_dir,'*.nii.gz']);

number = 1021;
for u =1:length(subjImageName)
    disp(u)
    disp(subjImageName(u).name)
    image = load_nii([file_image_dir,subjImageName(u).name]);
    label =  load_nii([file_label_dir,subjImageName(u).name]);
    
    mriVolume_image = squeeze(image.img);
    mriVolume_label = squeeze(label.img);
    
    theta = pi/2;
    for i =1:3
        %rotate in y axis
%         t = [cos(i*theta) 0 -sin(i*theta) 0
%             0 1 0 0
%             sin(i*theta) 0 cos(theta) 0
%             0 0 0 1]

        %rotate in z axis
        t=[cos(i*theta) sin(i*theta) 0 0
            -sin(i*theta) cos(i*theta) 0 0
            0 0 1 0
            0 0 0 1];
        tform = affine3d(t);
        mriVolumeRotated_image = imwarp(mriVolume_image,tform,'nearest');   %default: linear, should use nearest
        mriVolumeRotated_label = imwarp(mriVolume_label,tform,'nearest');
        
        axis =  unidrnd(3);
        mriVolumeRotated_image =int16( flip(mriVolumeRotated_image,axis));
        mriVolumeRotated_label = int16(flip(mriVolumeRotated_label,axis));
        %the original label value is 205,420,500,550,600,820,850

        mriVolumeRotated_label(mriVolumeRotated_label<200)=0;
        mriVolumeRotated_label(200<=mriVolumeRotated_label & mriVolumeRotated_label<=210)=205;
        mriVolumeRotated_label(210<mriVolumeRotated_label & mriVolumeRotated_label<415)=0;
        mriVolumeRotated_label(415<=mriVolumeRotated_label & mriVolumeRotated_label<=425)=420;
        mriVolumeRotated_label(425<mriVolumeRotated_label & mriVolumeRotated_label<495)=0;
        mriVolumeRotated_label(495<=mriVolumeRotated_label & mriVolumeRotated_label<=505)=500;
        mriVolumeRotated_label(505<mriVolumeRotated_label & mriVolumeRotated_label<545)=0;
        mriVolumeRotated_label(545<=mriVolumeRotated_label & mriVolumeRotated_label<=555)=550;
        mriVolumeRotated_label(555<mriVolumeRotated_label & mriVolumeRotated_label<595)=0;
        mriVolumeRotated_label(595<=mriVolumeRotated_label & mriVolumeRotated_label<=605)=600;
        mriVolumeRotated_label(605<mriVolumeRotated_label & mriVolumeRotated_label<815)=0;
        mriVolumeRotated_label(815<=mriVolumeRotated_label & mriVolumeRotated_label<=825)=820;
        mriVolumeRotated_label(825<mriVolumeRotated_label & mriVolumeRotated_label<845)=0;
        mriVolumeRotated_label(845<=mriVolumeRotated_label & mriVolumeRotated_label<=855)=850;
        mriVolumeRotated_label(mriVolumeRotated_label>855)=0;


%         % can use a single function to achieve this 
%          mriVolumeRotated_image = imrotate3(mriVolume_image,i*90,[0 0 1],'nearest','loose','FillValues',0);
%          mriVolumeRotated_label = imrotate3(mriVolume_label,i*90,[0 0 1],'nearest','loose','FillValues',0);
        %save the rotated data to nii.gz
        image_nii = make_nii(mriVolumeRotated_image);
        label_nii = make_nii(mriVolumeRotated_label);
        filename_image=[file_image_dir,'mr_train_',num2str(number),'.nii'];
        filename_label=[file_label_dir,'mr_train_',num2str(number),'.nii'];
        save_nii(image_nii,filename_image);
        save_nii(label_nii,filename_label);
        number = number +1;
    end
end