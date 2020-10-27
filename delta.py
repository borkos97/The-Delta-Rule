import json
import random


def readConfig(filename='config.json'):
    """
    The function reads the given parameters from the config.json file
    """
    with open(filename, 'r') as f:
        return json.load(f)


def preparePatterns(condition):
    return list(map(float, condition))


def loadData(name):
    """
    The function loads and preprocesses the .txt file
    """
    inputs = []
    outputs = []
    with open(name) as file:
        data = file.readlines()[2:]
        lines = map(str.split, data)
        for line in lines:
            inputs.append(preparePatterns(line[:-1]))
            outputs.append(float(line[-1]))
    length = len(inputs[0])
    return inputs, outputs, length


def randomWeights(patterns):
    """
    Function creating arrays with weight values
    """
    weights = []
    for i in range(patterns):
        weights.append(random.random())
    return weights


def train(inputs, outputs, epochs, learning_rate, weights):
    for i in range(epochs):
        for i in range(len(inputs)):
            single_inputs_line = inputs[i]
            single_output = outputs[i]
            y = calculateNeuronsOutputs(weights, single_inputs_line)
            weights = updateWeights(single_inputs_line, single_output, learning_rate, y, weights)
    return weights


def calculateNeuronsOutputs(weights, inputs):
    """
    Function that returns the final result
    """
    y = 0
    for i in range(len(weights)):
        y = y + weights[i] * inputs[i]
    return y


def updateWeights(inputs, outputs, learning_rate, y, weights):
    """
    Function that updates the weights with a formula
    """
    for i in range(len(weights)):
        weights[i] = weights[i] + learning_rate * (outputs - y) * inputs[i]
    return weights


final_results = []
params = readConfig()
inputs, outputs, length = loadData(params['filename'])
weights = randomWeights(length)
final_weights = train(inputs, outputs, params['epochs'], params['learning_rate'], weights)

for i in range(len(inputs)):
    final_results.append(calculateNeuronsOutputs(inputs[i], final_weights))
print("results", final_results, "\nweights", final_weights)