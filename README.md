# keras_siamese_networks
A siamese network model of keras, include a data generator for big-data-training

## Omniglot Dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/36079867-c94b19fe-0f7f-11e8-9ef8-6f017d214d43.png" alt="Omniglot Dataset"/>
</p>

The Omniglot dataset consists in 50 different alphabets, 30 used in a background set and 20 used in a evaluation set. Each alphabet has a number of characters from 14 to 55 different characters drawn by 20 different subjects, resulting in 20 105x105 images for each character. The background set should be used in training for hyper parameter tuning and feature learning, leaving the final results to the remaining 20 alphabets, never seen before by the models trained in the background set. Despite that this paper uses 40 background alphabets and 10 evaluation alphabets. 

### Code Details

There are two main files to run the code in this repo: 
- *model_train.py* that allows you to train a siamese network with a specific dataset. 
- *model_train_gen.py* that allows you to train a siamese network with a customization generator on a bigger dataset. 
- *base_network.py* bulid your base_network here.

Regarding the rest of the code:
- omniglot_preprocess is a script used to place the image in omniglot dataset to a specific directory. 
- data_augment is a script that can generate new image base on native image. If you want to train your network on your own datasets and don't have enough image, that would be useful.
- dump_data is used to create image pairs and labels then save them as numpy array in a pickle file.
- dump_data_gen same as above.But it save files for generator and there is almost no size limit of the image pairs data for train.
- generator is used by model_train_gen to generate image pairs for network training.

**Notes:**
- If you want to use your own dataset,you could put your image in dataset directory.Then you could run data_augment.py if you need augment your data or you could run dump_data.py to make a pickle file and use model_train.py to train your networks.Remember to change the path in the code.

### 代码简介

两个主要程序: 
- *model_train.py* 用于训练你的神经网络，需要先把图相对保存至pickle文件中，使用dump_data即可. 
- *model_train_gen.py* 可以用生成器训练更大的数据集，数据过多时无法全部读取到内存中，因此要用到生成器，运行之前要利用dump_data_gen.py进行数据预处理. 

其它程序:
- omniglot_preprocess.py 把omniglot dataset文件夹中的文件存储到对应目录中(omniglot processed). 
- data_augment.py 用于数据增强的脚本，如果你想在自己的图像数据集上训练神经网络，而图像数量不足，就可以用这个脚本生成新图像扩充数据集，自定义数据集按照格式存放在dataset文件夹中即可.
- dump_data.py 用omniglot processed文件夹中的图片制作图像对并记录标签，统一保存至一个pkl文件中，用于model_train.py的训练.
- dump_data_gen.py 同上，但是可以在更大的数据集上利用model_train_gen.py训练神经网络，当你发现用dump_data.py保存的文件超过10GB时可以考虑使用生成器方式进行训练。某种程度上可以组织网络过拟合，适应更多分类的任务。

**Notes:**
- 如果你想在自己的数据集上训练网络，你可以把图片按照格式保存在dataset或Omnitlog processed文件夹中，每个子文件夹代表一个图像种类，如/dataset/01和/dataset/02就分别存放两个种类的图片，可以根据需要自行替换文件夹中原有图片。替换后只需要修改用到的程序开头的部分路径设置或图像格式设置即可。

