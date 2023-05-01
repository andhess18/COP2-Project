#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using cv::Mat;

class Layer {
public:
    virtual void forward(const Mat& input_data) = 0;
    virtual Mat getActivationDerivative(const Mat& inputData) = 0;

    void gradientDescent(double learningRate);

    const Mat& getWeights() const { return weights; }
    void setWeights(const Mat& newWeights) { weights = newWeights; }

    const Mat& getWeightGradients() const { return weightGradients; }
    void setWeightGradients(const Mat& newWeightGradients) { weightGradients = newWeightGradients; }

    const Mat& getBiases() const { return biases; }
    void setBiases(const Mat& newBiases) { weights = newBiases; }

    const Mat& getBiasGradients() const { return biasGradients; }
    void setBiasGradients(const Mat& newBiasGradients) { biasGradients = newBiasGradients; }

    virtual void setInputData(const Mat& inputData) { input = inputData.clone(); }
    virtual const Mat& getOutput() const { return outputData; }

protected:
    Mat input;
    Mat outputData;
    Mat weights;
    Mat biases;
    Mat weightGradients;
    Mat biasGradients;
};

#endif