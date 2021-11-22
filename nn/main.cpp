#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include "NeuralNet.h"

int main()
{
	std::vector<std::vector<double>> inputData;
	std::vector<std::vector<double>> trainData;
	inputData.push_back({ 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 });
	inputData.push_back({ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 });
	inputData.push_back({ 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1 });

	std::shared_ptr<ActivationFunction> step = std::shared_ptr<ActivationFunction>(new AFStep());

	int inputCount = 20;
	HopfieldNet nn(inputCount);

	nn.setActivationFunction(nn.layers[0], step);
	nn.setActivationFunction(nn.layers[1], step);

	std::cout << "TRAIN" << std::endl;

	for (int i = 0; i < inputData.size(); i++) {
		nn.trainNetwork(inputData[i]);
	}

	std::vector<double> outputData;

	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		outputData = nn.outputVector();
		std::cout << "#" << i + 1 << std::endl;
		for (int k = 0; k < outputData.size(); k++) {
			std::cout << std::to_string(outputData[k]).substr(0, 4) << "; ";
		}
		std::cout << std::endl;
	}

	for (int k = 0; k < inputData.size(); k++) {
		for (int i = 0; i < 3; i++) {
			int ind = rand() % (inputData[0].size() - 1);
			inputData[k][ind] = (inputData[k][ind] == 1) ? -1 : 1;
		}
	}

	std::cout << "\nOutput" << std::endl;

	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		outputData = nn.outputVector();
		std::cout << "Random#" << i + 1 << std::endl;
		for (int k = 0; k < inputData[i].size(); k++) {
			std::cout << std::to_string(inputData[i][k]).substr(0, 4) << "; ";
		}
		std::cout << std::endl;
		std::cout << "Output#" << i + 1 << std::endl;
		for (int k = 0; k < outputData.size(); k++) {
			std::cout << std::to_string(outputData[k]).substr(0, 4) << "; ";
		}
		std::cout << std::endl;
	}
	return 0;
}