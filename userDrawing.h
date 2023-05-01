#ifndef USERDRAWING_H
#define USERDRAWING_H


#include <opencv2/opencv.hpp>


using cv::Mat;

void getuserImage(NeuralNetwork& neuralNet);

Mat processUserImage(Mat& curentuserImage);


#endif


