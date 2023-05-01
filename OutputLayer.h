#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "Layer.h"
#include <random>

class OutputLayer : public Layer {
public:
    OutputLayer(int inputSize, int outputSize);
    void forward(const Mat& input_data) override;
    Mat getActivationDerivative(const Mat& inputData) override;

private:
    void softmax(Mat& input);
};

#endif