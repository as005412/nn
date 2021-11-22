#include "NeuralNet.h"

Layer::Layer(int count) {
	for (int i = 0; i < count; i++)
		neurons.push_back(Neuron());
}

Layer::Layer(int count, Layer& nextLayer) {
	for (int i = 0; i < count; i++)
		neurons.push_back(Neuron());
	for (int i = 0; i < count * nextLayer.neurons.size(); i++)
		weights.push_back((rand() % 100) / 100.0);
}

double Layer::getWeight(int i, int j) {
	return weights[i * (weights.size() / neurons.size()) + j];
}

void Layer::setWeight(int i, int j, double value) {
	weights[i * (weights.size() / neurons.size()) + j] = value;
}

VectorQuantization::VectorQuantization(int inputSize, int codeVectorCount) {
	layers.insert(layers.begin(), Layer(inputSize)); // input layer
	for (int i = 0; i < codeVectorCount; i++) {
		layers.push_back(Layer(inputSize, layers[0]));
	}
}

void VectorQuantization::input(std::vector<double>& inputData) {
	for (int i = 0; i < layers[0].neurons.size(); i++)
	{
		layers[0].neurons[i].output = inputData[i];
	}
} 

void VectorQuantization::setCodeVector(std::vector<std::vector<double>>& value) {
	for (int l = 1; l < layers.size(); l++)
		for (int i = 0; i < layers[l].neurons.size(); i++)
			layers[l].neurons[i].output = value[layers.size() - l - 1][i];
}

void VectorQuantization::train() {
	int time = 0;
	while (time <= epochCount)
	{
		int winVectorIndex = 0;
		normVector.clear();
		for (int i = 0; i < layers.size() - 1; i++)
			normVector.push_back(0.0);
		double step = 0.1 * exp(-(double)time / 1000.0);
		for (int l = 1; l < layers.size(); l++)
		{
			double D = 0.0;
			for (int j = 0; j < layers[l].neurons.size(); j++)
				for (int i = 0; i < layers[0].neurons.size(); i++) {
					D += pow(layers[0].neurons[i].output - layers[l].getWeight(j, i), 2);
				}
			normVector[l - 1] = sqrt(D);
		}
		double minD = normVector[0];
		for (int i = 1; i < normVector.size(); i++)
			if (normVector[i] < minD) {
				minD = normVector[i];
				winVectorIndex = i;
			}
		winVectorIndex++;
		for (int j = 0; j < layers[winVectorIndex].neurons.size(); j++)
			for (int i = 0; i < layers[0].neurons.size(); i++) {
				double oldWeight = layers[winVectorIndex].getWeight(j, i);
				double newWeight = oldWeight + step * (layers[0].neurons[i].output - oldWeight);
				layers[winVectorIndex].setWeight(j, i, newWeight);
			}
		time += 1;
	}
}

void VectorQuantization::calculate() {
	int winVectorIndex = 0;
	normVector.clear();
	normVector.resize(layers.size() - 1);
	for (int l = 1; l < layers.size(); l++)
	{
		double D = 0.0;
		for (int j = 0; j < layers[l].neurons.size(); j++)
			for (int i = 0; i < layers[0].neurons.size(); i++) {
				D += pow(layers[0].neurons[i].output - layers[l].getWeight(j, i), 2);
			}
		normVector[l - 1] = sqrt(D);
	}
	double minD = normVector[0];
	for (int i = 1; i < normVector.size(); i++)
		if (normVector[i] < minD) {
			minD = normVector[i];
			winVectorIndex = i;
		}
	winVector = winVectorIndex + 1;
}

std::vector<double> VectorQuantization::outputVector() {
	std::vector<double> out;
	calculate();
	for (int i = 0; i < layers[winVector].neurons.size(); i++)
		out.push_back(layers[winVector].neurons[i].output);
	return out;
}