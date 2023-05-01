#include "HiddenLayer.h"
#include <cmath>
#include <random>

HiddenLayer::HiddenLayer(int inputSize, int outputSize) {
    // Initialize random number generator with mean 0 and standard deviation 0.1
    std::normal_distribution<double> dist(0.0, 0.1);
    std::random_device rd;
    std::mt19937 gen(rd());

    weights = Mat(inputSize, outputSize, CV_64F);
    biases = Mat(1, outputSize, CV_64F);

    for (int row = 0; row < inputSize; ++row) {
        for (int col = 0; col < outputSize; ++col) {
            weights.at<double>(row, col) = dist(gen);
        }
    }

    for (int col = 0; col < outputSize; ++col) {
        biases.at<double>(0, col) = dist(gen);
    }
}

double HiddenLayer::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void HiddenLayer::forward(const Mat& input_data) {
    setInputData(input_data);
    Mat preActivation = input * weights + biases;
    outputData = preActivation.clone();
    for (int col = 0; col < preActivation.cols; ++col) {
        outputData.at<double>(0, col) = sigmoid(preActivation.at<double>(0, col));
    }
}

double HiddenLayer::sigmoidDerivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

Mat HiddenLayer::getActivationDerivative(const Mat& inputData) {
    Mat result = inputData.clone();
    for (int col = 0; col < inputData.cols; ++col) {
        result.at<double>(0, col) = sigmoidDerivative(inputData.at<double>(0, col));
    }
    return result;
}

