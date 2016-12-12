# Gender classification, a SVM ver VGG FACE features

### Pierre de Tournemire

## Requirements
### Classification only
- OpenCV
- Caffe (libcaffe.so) https://github.com/BVLC/caffe/pull/1667 for help
- CMake

### Full Training
- Cudnn enabled Caffe with python Interface
- Python 2
   -- Numpy, H5py, Scikit-learn

## Run Classification (C++)

### Compilation

``` 
cmake .
make
```

### Usage
This classify a single image, with a learned SVM classifier, using 'fc7' features computed with Caffe (CPU ONLY). Put *vgg_face_caffe/* (from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) in the same directory and run :
``` 
export GLOG_minloglevel=2
./caffeinated_application <path_to_image_file>
```
For instance try `./caffeinated_application vgg_face_caffe/ak.png`
## Train SVM (Python 2, Caffe with GPU) **OPTIONNAL**

### Features computation
The first script aptly named *extract_features.py* computes the features of all images in celebA, extract the annotations and put them both in a .hdf5 database for later use. Without a modern GPU it is not practical, but we don't need the whole dataset...

To run download *img_align_celeba/* and *list_attr_celeba.txt* from the celebA dataset and do `python extract_features.py <pycaffe path> <img_align_celeba path> <list_attr_celeba.txt path> [max_number]` where <pycaffe> path is the path towards the python folder in your caffe folder.

### SVM classification
The second script *svm.py* trains and serialises a SVM classifier on the obtained features. As is, it uses only 2000 for training and a 3 x10000 for testing. **NOTE** It won't run if there is less than 40000 features ! 

Make sure that *celebfeatures.hdf5* is present and run `python svm.py`, this prints the correct classification rate and overwrites *svm.txt* with a serialised SVM vector and parameters.
