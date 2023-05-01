#include "Layer.h"

void Layer::gradientDescent(double learningRate) {
    weights -= learningRate * weightGradients;
    biases -= learningRate * biasGradients;
}


