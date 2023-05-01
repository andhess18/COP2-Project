#ifndef ABSTRACTNEURALNETWORK_H
#define ABSTRACTNEURALNETWORK_H

#include "INeuralNetwork.h"
#include <vector>
#include <opencv2/opencv.hpp>

using cv::Mat;
using std::vector;

class AbstractNeuralNetwork : public INeuralNetwork {
public:
    AbstractNeuralNetwork(int inputNodes, const vector<int>& hiddenNodes, int outputNodes)
        : inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes) {}

    virtual void train(const vector<Mat>& inputImages, const vector<uint8_t>& labels, int epochs, double learningRate) = 0;

protected:
    int inputNodes;
    int outputNodes;
    vector<int> hiddenNodes;
    vector<Mat> weights;
    vector<Mat> biases;
};

#endif






