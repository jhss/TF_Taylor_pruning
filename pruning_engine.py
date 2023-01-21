"""
This code is modified by Juhong from NVlab's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/NVlabs/Taylor_pruning/
"""
import tensorflow as tf
import numpy as np

# (1) Add gate layer to the pre-trained network.
# (2) Estimate pruning score based on the square of gradient for each unit in the gate layer.
# (3) If the score of a unit in the gate is less than threshold,
#     assigns the value of unit to zero to prune the weight connected with the gate unit.
class GradientPruning(object):
    def __init__(self, model, parameters):
        self.model = model
        self.gradients = None
        self.parameters = parameters

        self.pruned_neurons = 0  # store the number of pruned neurons
        self.prune_per_iteration = 100

        self.pruning_threshold = 0.0

        self.score_per_gate = [ [0.0 for unit in range(gate.shape[-1])] for gate in self.parameters]
        self.accumulated_magnitude = [ [] for _ in range(len(self.parameters))]

        self.cur_iterations = 0


    def step(self, gradients):
        '''
        (1) estimate_pruning_score: Calculate the square of gradients, and accumulate it into 'prune_network_accumulate'.
        (2) estimate_pruning_threshold: sort unit in the gate by its score, concatenate all units a
        '''

        self.gradients = gradients

        self.estimate_pruning_score()
        self.estimate_pruning_threshold()
        self.pruning_step()

        all_neurons, self.pruned_neurons = self._count_number_of_neurons()

        return all_neurons, self.pruned_neurons

    def estimate_pruning_score(self):

        for gate_idx, (parameter, gradient) in enumerate(zip(self.parameters, self.gradients)):

            gate_pruning_score = tf.pow(tf.multiply(parameter, gradient), 2)

            mult = 3.0
            if gate_idx == 1: mult = 4.0
            elif gate_idx == 2: mult = 6.0

            gate_pruning_score /= mult

            if self.cur_iterations == 0:
                self.accumulated_magnitude[gate_idx] = gate_pruning_score
            else:
                self.accumulated_magnitude[gate_idx] += gate_pruning_score

        self.cur_iterations += 1

    def estimate_pruning_threshold(self):

        for gate_idx, gate in enumerate(self.parameters):

            cur_gate_scores = self.accumulated_magnitude[gate_idx] / self.cur_iterations
            n_units = gate.shape[-1]

            for unit_idx in range(n_units):
                unit_score = cur_gate_scores[unit_idx].numpy().item()
                self.score_per_gate[gate_idx][unit_idx] =  unit_score * gate[unit_idx]

        flattend_scores = np.asarray([score for scores in self.score_per_gate for score in scores]).reshape(-1)

        # For each pruning step, additional neurons by 'prune_per_iteration' are removed.
        threshold_neuron_idx = self.pruned_neurons + self.prune_per_iteration - 1

        # adaptively estimate threshold given a number of neurons to be removed
        self.pruning_threshold = np.sort(flattend_scores)[threshold_neuron_idx]

    def pruning_step(self):

        for gate_idx, parameter in enumerate(self.parameters):
            cur_layer_weight = self.model.get_layer(f'gate_layer_{gate_idx}').get_weights()[0]

            for unit_idx, magnitude in enumerate(self.score_per_gate[gate_idx]):
                if magnitude <= self.pruning_threshold:
                    # do actual pruning
                    cur_layer_weight[unit_idx] = 0.0
                    self.parameters[gate_idx][unit_idx].assign(0.0)

            self.model.get_layer(f'gate_layer_{gate_idx}').set_weights([cur_layer_weight])

    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        all_neurons = 0
        remain_neurons = 0
        for idx, parameter in enumerate(self.parameters):

            cur_num_params = parameter.shape[-1]
            all_neurons += cur_num_params
            for unit in range(cur_num_params):
                statistics = parameter[unit]

                if statistics > 0.0:
                    remain_neurons += 1

        pruned_neurons = all_neurons - remain_neurons

        return all_neurons, pruned_neurons

def select_pruning_parameters(model):

    candidate_pruning_parameters = []
    param_name = []
    for layer in model.layers:
        if 'gate' in layer.name:
            candidate_pruning_parameters.append((layer.name, layer.weights[0]))

    candidate_pruning_parameters = sorted(candidate_pruning_parameters, key = lambda x: int(x[0].split("_")[-1]))
    candidate_pruning_parameters = [val for key, val in candidate_pruning_parameters]

    return candidate_pruning_parameters
