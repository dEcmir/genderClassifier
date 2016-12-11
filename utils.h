// Main script, inspired from features extratcion in caffe

#include <string>
#include "caffe/blob.hpp"


using caffe::Blob;
using std::string;

int svm_classifier(const string & imagePath, const float* x , int size);

void load_image(const string & imagePath, caffe::Blob<float>* input_layer);
