import sys
import numpy as np
import random
import json
from constants import NNConfig

class JsonHandler:
    def __init__(self, file_name, input_count, layer_arr):
        self.file_name = file_name
        self.input_count = input_count
        self.layer_arr = layer_arr
    
    def read_from_json(self):
        try:
            with open(self.file_name, "r") as file:
                data = json.load(file)
        except:
            data = {"layers": []}

        self.inspect_json(data)
        return data
        
    def inspect_json(self, data): # inspect the json file for the correct number of layers, weights, and biases
        # check number layers
        if(len(self.layer_arr) > len(data["layers"])):
            inp = input("WARNING: The requested neural network structure has more layers than the original. This will cause the original to have the new layer(s) added automatically. Continue?(y/n)")
            if inp != "y":
                sys.exit("Session Aborted. The requested neural network structure has more layers than the original")
            
            extra_layers = len(self.layer_arr) - len(data["layers"])
            self.write_json_change_layers(data, self.layer_arr[-extra_layers:])
            self.read_from_json()
            return
        elif(len(self.layer_arr) < len(data["layers"])):
            inp = input("WARNING: The requested neural network structure has less layers than the original. This will cause the original to have the layer(s) REMOVED automatically. VALUABLE NEURAL NETWORK VALUES COULD BE LOST. Continue?(y/n)")
            if inp != "y":
                sys.exit("Session Aborted. The requested neural network structure has less layers than the original")

            layers_to_remove = len(data["layers"]) - len(self.layer_arr)
            self.write_json_change_layers(data, num_remove_layers=layers_to_remove)
            self.read_from_json()
            return
        # check number biases per layer
        indexes_to_update = []
        for i in range(len(data["layers"])):
            if(self.layer_arr[i] > len(data["layers"][i]["biases"])):
                inp = input("WARNING: The requested neural network structure has more neurons in layer " + str(i) + " than the original. This will cause the original to have the new layer(s) added automatically. Continue?(y/n)")
                if inp != "y":
                    sys.exit("Session Aborted. The requested neural network structure has more neurons in layer " + str(i) + " than the original")
                
                indexes_to_update.append(i)
            elif(self.layer_arr[i] < len(data["layers"][i]["biases"])):
                inp = input("WARNING: The requested neural network structure has less neurons in layer " + str(i) + " than the original. This will cause the original to have the neurons(s) REMOVED automatically. VALUABLE NEURAL NETWORK VALUES COULD BE LOST. Continue?(y/n)")
                if inp != "y":
                    sys.exit("Session Aborted. The requested neural network structure has less neurons in layer " + str(i) + " than the original")
                
                indexes_to_update.append(i)
        if indexes_to_update:
            self.write_json_change_neurons(data, indexes_to_update)
            self.read_from_json()
            return

    def write_json_change_layers(self, data, arr_layers = None, num_remove_layers = None): # adds or removes layers
        if(arr_layers is None and num_remove_layers is None): # user didn't select to add or remove layers
            sys.exit("Session Aborted. To change layer number, either arr_layers or num_remove_layers params must be filled")
        elif(arr_layers is not None and num_remove_layers is not None): # user did both (illegal)
            sys.exit("Session Aborted. Only add layers OR remove layers, not both")
        elif(arr_layers is not None): # add layers
            for nodes in arr_layers:
                weights_arr2d = []
                biases_arr = []
                if(len(data["layers"]) < 1): # if there are no hidden layers yet
                    prev_layer_nodes = self.input_count
                else:
                    prev_layer_nodes = len(data["layers"][-1]["biases"])
                for i in range(nodes):
                    biases_arr.append(0.0)
                    weights_arr = []
                    for j in range(prev_layer_nodes):
                        weights_arr.append(random.uniform(-NNConfig.SEED_WEIGHT_MAX, NNConfig.SEED_WEIGHT_MAX))
                    weights_arr2d.append(weights_arr)
                data["layers"].append({
                    "weights": weights_arr2d,
                    "biases": biases_arr
                })

        elif(num_remove_layers is not None): # remove layers
            for i in range(num_remove_layers):
                data["layers"].pop()

        with open(self.file_name, "w") as file: # FINALLY, add/remove from json
            json.dump(data, file, indent=4)

    def write_json_change_neurons(self, data, indexes_to_update):
        for i in indexes_to_update:
            layer = data["layers"][i]

            current_nodes = len(layer["biases"])
            target_nodes = self.layer_arr[i]

            prev_nodes = self.input_count if i == 0 else len(data["layers"][i-1]["biases"])

            if target_nodes > current_nodes: # add neurons
                nodes_to_add = target_nodes - current_nodes

                for _ in range(nodes_to_add):
                    # add bias
                    layer["biases"].append(0.0)

                    # add weights for this neuron
                    new_weights = [random.uniform(-0.5, 0.5) for _ in range(prev_nodes)]
                    layer["weights"].append(new_weights)

            elif target_nodes < current_nodes: # remove neurons
                nodes_to_remove = current_nodes - target_nodes

                for _ in range(nodes_to_remove):
                    layer["biases"].pop()
                    layer["weights"].pop()

            # fix NEXT layer input size
            if i + 1 < len(data["layers"]):
                next_layer = data["layers"][i + 1]

                for neuron_weights in next_layer["weights"]:
                    if target_nodes > current_nodes:
                        # add new incoming weights
                        for _ in range(target_nodes - current_nodes):
                            neuron_weights.append(random.uniform(-0.5, 0.5))
                    elif target_nodes < current_nodes:
                        # remove extra weights
                        for _ in range(current_nodes - target_nodes):
                            neuron_weights.pop()

        # write back to file
        with open(self.file_name, "w") as file:
            json.dump(data, file, indent=4)