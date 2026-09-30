import sys
import numpy as np
import random
import json
from json_handler import JsonHandler
from constants import NNConfig

class NeuralNetwork:
    def __init__(self, file_name, hidden_arr, input_count = NNConfig.DEF_INPUT_NODES, output_count = NNConfig.DEF_OUTPUT_NODES):
        self.input_count = input_count # it is not reccomended to change this number
        self.hidden_arr = hidden_arr # an array where each index corresponds to a layer and each value corresponds to the number of neurons within that layer
        self.output_count = output_count # it is not reccomended to change this number
        self.layer_arr = hidden_arr + [output_count]

        self.file_name = file_name
        self.json_handler = JsonHandler(self.file_name, self.input_count, self.layer_arr)
        self.network = self.json_handler.read_from_json()

    def sigmoid(self, x): # helper function
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x): # helper function
        return max(0, x)
    
    def make_decision(self, nn, inputs): # returns the number of the index of the decision made
        prev_calc = inputs
        curr_calc = []
        # INCOMING: 3-LAYER FOR LOOP
        # HAVE FUN READING THIS SHITASS
        for l in range(len(nn["layers"])): # layers
            for cl in range(len(nn["layers"][l]["biases"])): # current layer
                node_calc = 0
                for pl in range(len(prev_calc)): # previous layer (for weights)
                    node_calc += prev_calc[pl] * nn["layers"][l]["weights"][cl][pl]

                node_calc += nn["layers"][l]["biases"][cl]
                if l != len(nn["layers"]) - 1:
                    node_calc = self.relu(node_calc)
                curr_calc.append(node_calc)

            prev_calc = curr_calc
            curr_calc = []
        
        greatest_val = prev_calc[0]
        greatest_val_index = 0
        for i in range(len(prev_calc)):
            if prev_calc[i] > greatest_val:
                greatest_val = prev_calc[i]
                greatest_val_index = i
        
        #print("outputs:", [round(x, 4) for x in prev_calc], "choice:", greatest_val_index)
        return greatest_val_index

    def mutate(self, nn):
        for l in range(len(nn["layers"])): # layers
            for b in range(len(nn["layers"][l]["biases"])): # biases
                if random.randint(1, NNConfig.BIAS_MUTATION_CHANCE) == 1:
                    nn["layers"][l]["biases"][b] += random.uniform(-NNConfig.BIAS_MUTATION_STRENGTH, NNConfig.BIAS_MUTATION_STRENGTH)
                for pl in range(len(nn["layers"][l]["weights"][b])): # previous layer (for weights)
                    if random.randint(1, NNConfig.WEIGHT_MUTATION_CHANCE) == 1:
                        nn["layers"][l]["weights"][b][pl] += random.uniform(-NNConfig.WEIGHT_MUTATION_STRENGTH, NNConfig.WEIGHT_MUTATION_STRENGTH)
        return nn
    
    def heavily_mutate(self, nn):
        for l in range(len(nn["layers"])): # layers
            for b in range(len(nn["layers"][l]["biases"])): # biases
                if random.randint(1, int(NNConfig.BIAS_MUTATION_CHANCE*NNConfig.HEAVY_MUTATION_CONSISTENCY)) == 1:
                    nn["layers"][l]["biases"][b] += random.uniform(-NNConfig.BIAS_MUTATION_STRENGTH*NNConfig.HEAVY_MUTATION_STRENGTH, NNConfig.BIAS_MUTATION_STRENGTH*NNConfig.HEAVY_MUTATION_STRENGTH)
                for pl in range(len(nn["layers"][l]["weights"][b])): # previous layer (for weights)
                    if random.randint(1, int(NNConfig.WEIGHT_MUTATION_CHANCE*NNConfig.HEAVY_MUTATION_CONSISTENCY)) == 1:
                        nn["layers"][l]["weights"][b][pl] += random.uniform(-NNConfig.WEIGHT_MUTATION_STRENGTH*NNConfig.HEAVY_MUTATION_STRENGTH, NNConfig.WEIGHT_MUTATION_STRENGTH*NNConfig.HEAVY_MUTATION_STRENGTH)
        return nn
