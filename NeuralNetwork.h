#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "AbstractNeuralNetwork.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"
#include "TrainingData.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

using std::vector;
using cv::Mat;

class NeuralNetwork : public AbstractNeuralNetwork {
public:
    NeuralNetwork(int inputNodes, const vector<int>& hiddenNodes, int outputNodes);

    void train(const vector<Mat>& inputImages, const vector<uint8_t>& labels, int epochs, double learningRate) override;
    int predict(const Mat& inputImage);

private:
    double crossEntropyLoss(const Mat& output, uint8_t label);
    void forward(const Mat& inputImage, vector<Mat>& layerOutputs);
    void backPropagation(const vector<Mat>& layerOutputs, const Mat& inputImage, uint8_t label, vector<Mat>& weightGradients, vector<Mat>& biasGradients);
    void gradientDescent(double learningRate, const vector<Mat>& weightGradients, const vector<Mat>& biasGradients);
    int numLayers;
    std::vector<std::shared_ptr<Layer>> layers;
    std::unique_ptr<InputLayer> inputLayer;
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
};

#endif