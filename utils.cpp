#ifndef UTILS_H_
#define UTILS_H_
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace cv;

void load_image(const string & imagePath, caffe::Blob<float>* input_layer )
{
    Mat img = imread(imagePath, -1);
    if(img.empty())
       std::cerr << "Unable to decode image " << imagePath << std::endl;

    std::vector<Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
       Mat channel(height, width, CV_32FC1, input_data);
       input_channels.push_back(channel);
       input_data += width * height;
}

	// Convert the input image to the input image format of the network.
	Mat sample;
	if (img.channels() == 4 && input_layer->channels() == 3)
		cvtColor(img, sample, COLOR_BGRA2BGR);
	else
		sample = img;

	Mat sample_resized;
	if (sample.size() != Size(width, height))
		resize(sample, sample_resized, Size(width, height));
	else
		sample_resized = sample;

	Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);
	Mat sample_normalized;
    // Mean Substraction
    // This operation will write the separate BGR planes directly to the
    // input layer of the network because it is wrapped by the Mat
    // objects in input_channels.
    split(sample_float,input_channels);
    subtract(input_channels[0], Mat(input_channels[0].size(), CV_32FC1, Scalar(104.00698793)) , input_channels[0]);
    subtract(input_channels[1], Mat(input_channels[0].size(), CV_32FC1, Scalar(116.66876762)), input_channels[1]);
    subtract(input_channels[2], Mat(input_channels[0].size(), CV_32FC1, Scalar(122.67891434)) , input_channels[2]);
}


int svm_classifier(const string & model_path, const float* x , int size){
     std::fstream f(model_path.c_str(), std::ios_base::in);

     // parsing step
     int vectors_nb, vector_size;
     f >> vectors_nb>> vector_size;

     float gamma, c0, tmp;
     f >> gamma;
     f >> c0;

     std::vector<float> alpha(vectors_nb);
     for(int i=0; i< vectors_nb; i++){
         f >> tmp;
         alpha[i] = tmp;
     }

     std::vector<std::vector<float> > vectors(vectors_nb, std::vector<float>(vector_size));


     for(int i=0; i< vectors_nb; i++){
         for(int j=0; j< vector_size; j++){
             f >> tmp;
             vectors[i][j] = tmp;
         }
     }

     // parsing is done, let's compute the classification of input x
     tmp = c0;
     float scal;
     for(int i=0; i< vectors_nb; i++){
         scal = 0;
         for(int j=0; j< vector_size; j++){
             scal += (vectors[i][j] * x[j]);
         }
         tmp += alpha[i] * std::tanh(gamma * scal);
     }
     if(tmp >= 0)
         return 1;
     else return -1;
}

#endif /* UTILS_H_ */
