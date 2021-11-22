#include <iostream>
#include <math.h>
#include <vector>
#include <string>

class Perceptron
{
public:
	double alpha = 0.05f;
	std::vector<double> neurons;
	std::vector<double> outputNeurons;
	std::vector<double> weights;
	std::vector<double> offsetNeurons;
	Perceptron(int neuronCount, int outputCount) {
		for (int i = 0; i < neuronCount; i++)
			neurons.push_back(0.0);
		for (int i = 0; i < outputCount; i++) {
			outputNeurons.push_back(0.0);
			offsetNeurons.push_back(0.0);
		}
		for (int i = 0; i < neuronCount * outputCount; i++)
			weights.push_back(0.0);
	}

	std::vector<double> output() { getResult(); return outputNeurons; }

	void trainNetwork(std::vector<double> outputData, std::vector<double> trainData){
		for (int k = 0; k < outputNeurons.size(); k++) {
			for (int i = 0; i < neurons.size(); i++) {
				int ind = i + k * outputNeurons.size();
				weights[ind] = weights[ind] - alpha * (outputData[k] - trainData[k]) * neurons[i];
			}
			offsetNeurons[k] = offsetNeurons[k] + alpha * (outputData[k] - trainData[k]);
		}
	}
	inline double activation(double value) { return value; }
private:
	void getResult() {
		for (int k = 0; k < outputNeurons.size(); k++)
		{
			double S = 0.0;
			for (int i = 0; i < neurons.size(); i++)
			{
				S += neurons[i] * weights[i + k * outputNeurons.size()];
			}
			S -= offsetNeurons[k];
			outputNeurons[k] = activation(S);
		}
	}
};

double sinFunction(double x) {
	int a = 2, b = 5;
	double d = 0.6;
	return a * sin(b * x) + d;
}

int main()
{
	std::vector<double> inputData;
	std::vector<double> outputData;
	std::vector<double> trainData;
	trainData.push_back(0.0);
	int inCount = 5, outCount = 1;
	Perceptron perceptron(inCount, outCount);
	std::cout << "TRAIN" << std::endl;
	double time = 0.0;
	for (int i = 0; i < inCount; i++) {
		time += 0.1;
		inputData.push_back(sinFunction(time));
	}
	for (int i = 0; i < 50; i++) {
		inputData.erase(inputData.begin());
		inputData.push_back(sinFunction(time));
		trainData[0] = sinFunction(time + 0.1);
		perceptron.neurons = inputData;
		outputData = perceptron.output();
		perceptron.trainNetwork(outputData, trainData);
		time += 0.1;
	}
	std::cout << "Training ends at " << time << " seconds." << std::endl;
	std::cout << "Attempt to prediction sin function from " << time << " seconds..." << std::endl;
	std::cout << "\tStandard\t\tOutput\t\t\tDeviation" << std::endl;
	outputData = perceptron.output();
	double E = 0;
	for (int i = 0; i < 15; i++) {
		inputData.erase(inputData.begin());
		inputData.push_back(outputData[0]);
		perceptron.neurons = inputData;
		outputData = perceptron.output();
		E = 0.5 * pow(outputData.at(0) - sinFunction(time + 0.1), 2);
		std::cout << "#" << i + 1 << "\t" << std::to_string(sinFunction(time + 0.1)).substr(0, 7) << "\t\t\t"
				  << std::to_string(outputData.at(0)).substr(0, 7) << "\t\t\t" << std::to_string(E).substr(0, 7) << std::endl;
		time += 0.1;
	}
	return 0;
}

