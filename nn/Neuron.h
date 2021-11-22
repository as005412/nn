#pragma once
#include "ActivationFunction.h"
class Neuron
{
public:
	Neuron() { }
	double input = 0;
	double output = 0;
	double propagationError = 0;
	ActivationFunction* function = nullptr;

	void addInput(double value);
	void active();
	double derivative();
};

