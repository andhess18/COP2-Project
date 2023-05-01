#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include "Layer.h"
#include <opencv2/opencv.hpp>
#include <random>

using cv::Mat;

class HiddenLayer : public Layer {
public:
    HiddenLayer(int inputSize, int outputSize);
    void forward(const Mat& input_data) override;
    Mat getActivationDerivative(const Mat& inputData) override;

private:
    double sigmoid(double x);
    double sigmoidDerivative(double x);
};

#endif