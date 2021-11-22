#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include "NeuralNet.h"

int main()
{
	std::vector<std::vector<double>> inputData;
	std::vector<std::vector<double>> trainData;
	inputData.push_back({ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 });
	inputData.push_back({ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
	inputData.push_back({ 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1 });

	int inputCount = 20;
	VectorQuantization nn(inputCount, inputData.size());

	std::cout << "TRAIN" << std::endl;

	nn.setCodeVector(inputData);
	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		nn.train();
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

	std::cout << "\nInverted Bits Train" << std::endl;

	for (int k = 0; k < inputData.size(); k++) {
		for (int i = 0; i < 3; i++) {
			int ind = rand() % (inputData[0].size() - 1);
			inputData[k][ind] = (inputData[k][ind] > 0) ? 0 : 1;
		}
	}

	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		nn.train();
		std::cout << "#" << i + 1 << std::endl;
		for (int k = 0; k < inputData[i].size(); k++) {
			std::cout << std::to_string(inputData[i][k]).substr(0, 4) << "; ";
		}
		std::cout << std::endl;
	}

	std::cout << "\nRandom bits output" << std::endl;
	inputData.clear();
	for (int k = 0; k < 5; k++) {
		std::vector<double> set;
		for (int i = 0; i < inputCount; i++) {
			set.push_back((rand()%100 < 50) ? 0 : 1);
		}
		inputData.push_back(set);
	}

	for (int i = 0; i < inputData.size(); i++) {
		nn.input(inputData[i]);
		outputData = nn.outputVector();
		std::cout << "\nR#" << i + 1 << std::endl;
		for (int k = 0; k < inputData[i].size(); k++) {
			std::cout << std::to_string(inputData[i][k]).substr(0, 4) << "; ";
		}
		std::cout << std::endl;
		std::cout << "O#" << i + 1 << std::endl;
		for (int k = 0; k < outputData.size(); k++) {
			std::cout << std::to_string(outputData[k]).substr(0, 4) << "; ";
		}
		std::cout << std::endl;
	}
	return 0;
}