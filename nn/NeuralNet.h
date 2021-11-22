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
};

class VectorQuantization
{
public:
	VectorQuantization(int inputSize, int codeVectorCount);
	std::vector<Layer> layers;
	int epochCount = 100;
	void input(std::vector<double>& inputData);
	void setCodeVector(std::vector<std::vector<double>>& value);
	void train();
	void calculate();
	std::vector<double> outputVector();
private:
	std::vector<double> normVector;
	int winVector = 0;
};

