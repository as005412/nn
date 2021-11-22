#include "NeuralNet.h"

Layer::Layer(int count) {
	for (int i = 0; i < count; i++)
		neurons.push_back(Neuron());
}

Layer::Layer(int count, Layer& nextLayer) {
	for (int i = 0; i < count; i++)
		neurons.push_back(Neuron());
	for (int i = 0; i < (count + 1) * nextLayer.neurons.size(); i++)
		weights.push_back(0.0);
}

double Layer::getWeight(int i, int j) {
	return weights[i * (weights.size() / neurons.size()) + j];
}

void Layer::setWeight(int i, int j, double value) {
	weights[i * (weights.size() / neurons.size()) + j] = value;
}

double Layer::getThreshold(int j) {
	return weights[neurons.size() + j];
}

void Layer::setThreshold(int j, double value) {
	weights[neurons.size() + j] = value;
}

HopfieldNet::HopfieldNet(int inputSize) {
	layers.insert(layers.begin(), Layer(inputSize));
	layers.insert(layers.begin(), Layer(inputSize, *layers.begin()));
}

void HopfieldNet::setActivationFunction(Layer& layer, std::shared_ptr<ActivationFunction>& function) {
	for (auto& neuron : layer.neurons)
		neuron.function = function.get();
}

void HopfieldNet::input(std::vector<double>& inputData) {
	for (int i = 0; i < layers[0].neurons.size(); i++)
	{
		layers[0].neurons[i].input = inputData[i];
		layers[0].neurons[i].active();
	}
}

void HopfieldNet::calculate() {
	for (int j = 0; j < layers[1].neurons.size(); j++)
	{
		double S = 0.0;
		for (int i = 0; i < layers[0].neurons.size(); i++)
		{
			if (i == j)
				continue;
			S += layers[0].neurons[i].output * layers[0].getWeight(i, j);
		}
		layers[1].neurons[j].input = S;
		layers[1].neurons[j].active();
	}
}

std::vector<double> HopfieldNet::outputVector() {
	std::vector<double> out;
	calculate();
	for (int i = 0; i < layers[1].neurons.size(); i++)
		out.push_back(layers[1].neurons[i].output);
	return out;
}

void HopfieldNet::trainNetwork(std::vector<double>& trainData) {
	for (int i = 0; i < trainData.size(); i++) {
		for (int j = 0; j < trainData.size(); j++) {
			layers[0].setWeight(i, j, layers[0].getWeight(i, j) + trainData[i] * trainData[j]);
		}
	}
}