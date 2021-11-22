#pragma once
#include "Neuron.h"
#include <iostream>
#include <vector>

class Layer
{
public:
	Layer(int count);
	Layer(int count, Layer& nextLayer);
	std::vector<Neuron> neurons;
	std::vector<double> weights;
	double getWeight(int i, int j);
	void setWeight(int i, int j, double value);
	double getThreshold(int j);
	void setThreshold(int j, double value);
};

class HopfieldNet
{
public:
	HopfieldNet(int inputSize);
	std::vector<Layer> layers;

	void input(std::vector<double>& inputData);
	void calculate();
	std::vector<double> outputVector();
	void trainNetwork(std::vector<double>& trainData);
	void setActivationFunction(Layer& layer, std::shared_ptr<ActivationFunction>& function);
};

