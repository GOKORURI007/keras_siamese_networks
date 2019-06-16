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

Regarding the rest of the code:
- omniglot_preprocess is a script used to place the image in omniglot dataset to a specific directory. 
- data_augment is a script that can generate new image base on native image. If you want to train your network on your own datasets and don't have enough image, that would be useful.
- dump_data is used to create image pairs and labels then save them as numpy array in a pickle file.
- dump_data_gen same as above.But it save files for generator and there is almost no size limit of the image pairs data for train.
- generator is used by model_train_gen to generate image pairs for network training.

**Notes:**
- If you want to use your own dataset,you could put your image in dataset directory.Then you could run data_augment.py if you need augment your data or you could run dump_data.py to make a pickle file and use model_train.py to train your networks.Remember to change the path in the code.
