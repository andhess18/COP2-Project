#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "Layer.h"
#include <random>

class InputLayer : public Layer {
public:
    InputLayer(int inputNodes, int outputNodes);
    void forward(const Mat& input_data) override;
    Mat getActivationDerivative(const Mat& inputData) override;

};

#endif