# Data to image

This repository contains some experiments in which we search for good ways of converting data to images capable of class distinction. For more information, please refer to this [partial report].

## Usage

We have three experiments repositories: `conv_iris/`, `conv_leaf/` and `conv_titanic/`. Let's choose `conv_iris/` as an example. 

* First, we have to create the images, which is done by the Mathematica notebook `create_images.nb`. 
* Then, we have to create two lists: one with the names of the images in our training set followed by the class (in the same line) and one similar for the test set. This is done by the script `make_names_list.py`.
* Now we have to create the leveldb data that Caffe will use for train and test. This is done by the script `create_data.sh`.
* Finally, we have to train and test our CNN. We have just to run the script `train.sh`. Of course we can play with the parameters in the Caffe models configurations files.

There is also a script `experiment.sh` that will do almost all steps above 50 times, which is a good way to train a model with multiple dataset partitions. This script will not create the images, so we must create them before calling it.

[partial report]:docs/report/report.pdf