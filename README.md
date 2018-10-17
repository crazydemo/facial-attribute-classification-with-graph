# facial-attribute-classification-with-graph
Facial attribute classification based on graph attention (tensorflow)

1.Datasets

CelebA(aligned & cropped version):http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

LFWA:http://vis-www.cs.umass.edu/lfw/

2.Pretrain Model

Alexnet with bn:converted from caffe model trained by Marcel Simon, Erik Rodner, Joachim Denzler.

You can download the caffe model here:

https://github.com/cvjena/cnn-models

We also provide the converted tensorflow model in our project, along with the converted resnet50 mdoel trained on VggFace2.

3.Data Augmentation

As the LFWA dataset is too small, we add some distortion to the training set. The distortion tool is from https://github.com/mdbloice/Augmentor
