#include "InputLayer.h"
#include <cmath>
#include <random>

InputLayer::InputLayer(int inputNodes, int outputNodes) {
    // Initialize random number generator with mean 0 and standard deviation 0.1
    std::normal_distribution<double> dist(0.0, 0.1);
    std::random_device rd;
    std::mt19937 gen(rd());

    weights = Mat(inputNodes, outputNodes, CV_64F);
    biases = Mat(1, outputNodes, CV_64F);

    for (int row = 0; row < inputNodes; ++row) {
        for (int col = 0; col < outputNodes; ++col) {
            weights.at<double>(row, col) = dist(gen);
        }
    }

    for (int col = 0; col < outputNodes; ++col) {
        biases.at<double>(0, col) = dist(gen);
    }
}

void InputLayer::forward(const Mat& input_data) {
    setInputData(input_data);
    outputData = input * weights + biases;
}

Mat InputLayer::getActivationDerivative(const Mat& inputData) {
    return Mat::eye(1, inputData.cols, CV_64F);
}

