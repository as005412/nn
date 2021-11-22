#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include "NeuralNet.h"

double numberByIndex(int ind) {
	if (ind == 0) return 6;
	if (ind == 1) return 5;
	if (ind == 2) return 8;
}

int main()
{
	std::vector<std::vector<double>> inputData;
	std::vector<std::vector<double>> trainData;
	inputData.push_back({ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
	trainData.push_back({ 1, 0, 0 });
	inputData.push_back({ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
	trainData.push_back({ 0, 1, 0 });
	inputData.push_back({ 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1 });
	trainData.push_back({ 0, 0, 1 });

	std::shared_ptr<ActivationFunction> sigmoid = std::shared_ptr<ActivationFunction>(new AFSigmoid());
	std::shared_ptr<ActivationFunction> linear = std::shared_ptr<ActivationFunction>(new AFLinear());

	int hiddenCount = 1, inputCount = 20, outputCount = 3;
	NeuralNet nn(inputCount, hiddenCount, 7, outputCount);

	nn.setActivationFunction(nn.layers[0], linear);
	for (size_t i = 0; i < hiddenCount; i++)
		nn.setActivationFunction(nn.layers[i + 1], sigmoid);
	nn.setActivationFunction(nn.layers[nn.layers.size() - 1], sigmoid);

	std::cout << "TRAIN" << std::endl;
	double td = 200.0;
	while (td >= 0.001)
	{
		td = 0;
		for (int i = 0; i < inputData.size(); i++) {
			nn.input(inputData[i]);
			nn.trainNetwork(trainData[i]);
			td += nn.deviation(trainData[i]);
		}
	}
	std::cout << "Total deviation: " << td << std::endl;
	std::cout << "\nTrain images Output" << std::endl;

	std::vector<double> outputData;

	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		outputData = nn.outputVector();
		for (int k = 0; k < outputData.size(); k++) {
			if (round(outputData[k]) > 0)
			{
				std::cout << "\nInput#" << i + 1 << ": ";
				for (int q = 0; q < inputData[i].size(); q++) {
					std::cout << std::to_string(inputData[i][q]).substr(0, 1) << "; ";
				}
				std::cout << "\nNetwork  output#" << i + 1 << ": " << std::to_string(numberByIndex(k)).substr(0, 1) << std::endl;
			}
		}
	}
	std::cout << "\nRandom images Output" << std::endl;
	inputData.clear();
	for (int i = 0; i < 6; i++) {
		std::vector<double> set;
		for (int k = 0; k < 20; k++)
			set.push_back((rand() % 100 < 50) ? 0 : 1);
		inputData.push_back(set);
	}

	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		outputData = nn.outputVector();
		for (int k = 0; k < outputData.size(); k++) {
			if (round(outputData[k]) > 0)
			{
				std::cout << "\nInput#" << i + 1 << ": ";
				for (int q = 0; q < inputData[i].size(); q++) {
					std::cout << std::to_string(inputData[i][q]).substr(0, 1) << "; ";
				}
				std::cout << "\nNetwork  output#" << i + 1 << ": " << std::to_string(numberByIndex(k)).substr(0, 1) << std::endl;
			}
		}
	}
	return 0;
}