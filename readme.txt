此版本部分代码还没进行优化。

使用说明：
1、由于代码还未优化，对于原始的nii文件，需要现将其分割成64*64*64的patch存储起来，程序直接读取存储好的patch文件。
2、由于使用的数据集比较少，需要对数据进行data augmentation。先用matlab文件data_augmentation将原始的nii图像文件进行翻转，旋转扩大数据集合，并将扩大之后的文件保存起来。
3、fFindImageBoundaryCoordinate3D.py文件 是根据label的值将原始图像中的boundingbox分割出来，保证分割出来的patch都能包括一定的label区域。
4、prepare_data.py 里面提供函数，可以将原始nii图像的boundingbox存储起来，并分割成patch存储起来。
5、GetData_new.py 提供函数给网络不断乱序feed patch，以及顺序的feed patch。
6、U_net_3D_multiLabels.py 网络的主体函数，可以训练时断点保存以及tensorBoard查看。Output文件夹 存储网络的参数文件，可以使用里面的断点文件，继续没有结束的训练。 log是tensorBoard需要的文件的存储目录。可以使用Output文件夹中的已经训练好的参数，分割http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/  中的MRI 心脏
7、U_net_3D_Test_multiLabels.py文件是 根据训练好的网络，来生成whole heart 的label。同样需要先将原始nii分割成patch。
8、U_net_3D_Test_multiLabels.py 会将所有的patch对应的label存储下来，下面就需要将patch重新拼接起来。
9、make_whole_multiLabels.py 根据阈值判断，将8个label分别保存成8个nii文件，里面每一个文件对应heart的一个label
10.asemble.py则是根据majority voting的原则，将8个label合并成一个whole heart，同时使用了连通性判断，只保留每一个label的最大的部分，去除掉小的islands。