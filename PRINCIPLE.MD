# GENDER CLASSIFICATION USING A SVM ON VGG_FACE FEATURES
--------------------------------------------------------

### Pierre de Tournemire

## Method
I chose a simple but well performing method : using deep CNN features to feed a SVM. I chose the 'fc7' layer from the VGG_FACE network. The last layer basically contains a probability map of the face localisation which is rather irrelevant to the gender, but the previous layer should contain more interesting face features. The SVM uses a sigmoid kernel so it is somewhat analoguous to train a single layer on top of the 'fc7' one.

I trained the SVM on only 2000 feature maps and tested on 100000 with an accuracy around 0.97

## Implementation details

The code is in two parts : 
- a C++ Proof of concept which takes a single image computes its features with Caffe (CPU only) and then classify it with the trained SVM.
- two python scripts that do the heavy lifting through Caffe (GPU) and Scikit-learn

The first one works alone with the caffe models and the svm.txt file which is a simple serialisation of the support vectors.

