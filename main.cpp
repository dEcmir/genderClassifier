// Main script, inspired from features extratcion in caffe
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"

#include "caffe/proto/caffe.pb.h"

#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "utils.h"


using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;





int main(int argc, char** argv) {
    if(argc != 2){
        std::cout << "Missing filepath as argument" << std::endl;
        return -1;
    }

    // script takes as only argument the image path
    std::string image_path(argv[1]);
    // here we use GPU only mode
    Caffe::set_mode(Caffe::CPU);

    //Net creation
    std::string pretrained_binary_proto("vgg_face_caffe/VGG_FACE.caffemodel");
    std::string feature_extraction_proto("vgg_face_caffe/VGG_FACE_deploy.prototxt");
    boost::shared_ptr<Net<float> > feature_extraction_net(
    new Net<float>(feature_extraction_proto, caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

    // Load image using util function
    load_image(image_path, feature_extraction_net->input_blobs()[0]);

    // actual feature computation
    feature_extraction_net->Forward();

    const boost::shared_ptr<Blob<float> > feature_blob =
        feature_extraction_net->blob_by_name("fc7");

    const float* feature_blob_data;

    feature_blob_data = feature_blob->cpu_data();

    // feature map is classified thanks to the SVM clasifier
    int sex = svm_classifier("svm.txt", feature_blob_data , feature_blob->channels());

    // output print
    std::cout << std:: endl << std:: endl << std:: endl << std:: endl;
    std::cout << "\tImage " << image_path << " is classified as ";
    if(sex ==1)
       std::cout << "male." << std::endl;
     else std::cout << "female." << std::endl;
    return 1;


}
