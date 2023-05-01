#include "OutputLayer.h"
#include <cmath>
#include <random>
#include <iostream>

OutputLayer::OutputLayer(int inputSize, int outputSize) {
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

// Softmax function: Applies the softmax activation function to the input matrix
void OutputLayer::softmax(Mat& input) {
    double max_val = -DBL_MAX;
    for (int col = 0; col < input.cols; ++col) {
        max_val = std::max(max_val, input.at<double>(0, col));
    }

    double exp_sum = 0.0;
    for (int col = 0; col < input.cols; ++col) {
        input.at<double>(0, col) = std::exp(input.at<double>(0, col) - max_val);
        exp_sum += input.at<double>(0, col);
    }

    for (int col = 0; col < input.cols; ++col) {
        input.at<double>(0, col) /= exp_sum;
    }
}

// Forward function: Computes the forward pass through the output layer
void OutputLayer::forward(const Mat& input_data) {
    setInputData(input_data);
    outputData = input * weights + biases;
    softmax(outputData);
}

// Activation derivative function: Computes the derivative of the softmax activation function.
Mat OutputLayer::getActivationDerivative(const Mat& inputData) {
    Mat result = inputData.clone();
    for (int col = 0; col < inputData.cols; ++col) {
        double softmax_value = inputData.at<double>(0, col);
        result.at<double>(0, col) = softmax_value * (1 - softmax_value);
    }
    return result;
}