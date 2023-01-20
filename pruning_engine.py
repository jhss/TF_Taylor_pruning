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

        self.prune_network_criteria = list()
        self.prune_network_accumulate = {"by_layer": list(), "averaged": list()}

        self.cur_iterations = 0
        self.group_size = 1

        for parameter in self.parameters:
            n_units = parameter.shape[-1]
            cur_layer_criteria = [0.0 for unit in range(n_units)]
            self.prune_network_criteria.append(cur_layer_criteria)

            # 각 key별로 layer 개수만큼 list 추가
            for key in self.prune_network_accumulate.keys():
                self.prune_network_accumulate[key].append(list())

    def step(self, gradients):
        '''
        (1) estimate_pruning_score: Calculate the square of gradients, and accumulate it into 'prune_network_accumulate'.
        (2) compute_saliency: add all score of the units in the same weight, and select the weight to be pruned.
        '''

        self.gradients = gradients

        self.estimate_pruning_score()
        self.compute_saliency()

        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.pruned_neurons = all_neuron_units - neuron_units

        return self.pruned_neurons, all_neuron_units

    def estimate_pruning_score(self):

        for dim_idx, (parameter, gradient) in enumerate(zip(self.parameters, self.gradients)):

            unit_pruning_score = tf.pow(tf.multiply(parameter, gradient), 2)

            mult = 3.0
            if unit_idx == 1: mult = 4.0
            elif unit_idx == 2: mult = 6.0

            unit_pruning_score /= mult

            if self.cur_iterations == 0:
                self.prune_network_accumulate['by_layer'][dim_idx] = unit_pruning_score
            else:
                self.prune_network_accumulate['by_layer'][dim_idx] += unit_pruning_score

        self.cur_iterations += 1


    # list_criteria_per_layer에 있는 layer를 하나의 group으로 묶는다.
    # 하나의 layer에 있는 모든 뉴런의 score를 더해서 layer마다 저장한다.
    # [ [(n1, score), (n2, score), ... ], [ ]] -> [[layer1, score], [layer2, score], ... ]
    def group_criteria(self, list_criteria_per_layer, group_size = 1):

        groups = list()
        for layer in list_criteria_per_layer:
            layer_groups = list()
            indices = np.argsort(layer)

            for group_id in range(int(np.ceil(len(layer)/group_size))):
                current_group = slice(group_id*group_size, min((group_id+1)*group_size, len(layer)))
                values = [layer[ind] for ind in indices[current_group]]
                group = [indices[current_group], sum(values)]
                layer_groups.append(group)
            groups.append(layer_groups)

        return groups

    def compute_saliency(self):

        for layer, parameter in enumerate(self.parameters):

            contribution = self.prune_network_accumulate["by_layer"][layer] / self.cur_iterations
            # [CHECK] whether use momentum or not
            # use momentum to accumulate criteria over several pruning iterations.
            # weighted average between previous average and current contribution.
            # 각 layer마다 계산된 criteria를 할당한다.
            self.prune_network_accumulate["averaged"][layer] = contribution
            current_layer = self.prune_network_accumulate["averaged"][layer]

            n_units = parameter.shape[-1]

            for unit in range(n_units):
                criterion_now = current_layer[unit].numpy().item()
                self.prune_network_criteria[layer][unit] =  criterion_now * parameter[unit]

        '''
        각 뉴런마다 할당한 contribution을 기반으로, group의 contribution을 계산한다.

        [CHECK] group criteria가 어떻게 되는지 확인해봐야함
        '''
        # create groups per layer

        groups = self.group_criteria(self.prune_network_criteria, group_size = self.group_size)

         # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)//self.group_size - 1

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]


        for layer, parameter in enumerate(self.parameters):
            cur_layer_weight = self.model.get_layer(f'gate_layer_{layer}').get_weights()[0]

            for group in groups[layer]:
                if group[1] <= threshold_now:
                    for unit in group[0]:
                        # do actual pruning
                        cur_layer_weight[...,unit] = 0.0
                        self.parameters[layer][unit].assign(0.0)
            self.model.get_layer(f'gate_layer_{layer}').set_weights([cur_layer_weight])

    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        all_neuron_units = 0
        neuron_units = 0
        for idx, parameter in enumerate(self.parameters):

            cur_num_params = parameter.shape[-1]
            all_neuron_units += cur_num_params
            for unit in range(cur_num_params):
                statistics = parameter[unit]

                if statistics > 0.0:
                    neuron_units += 1

        return all_neuron_units, neuron_units

def select_pruning_parameters(model):

    candidate_pruning_parameters = []
    param_name = []
    for layer in model.layers:
        if 'gate' in layer.name:
            candidate_pruning_parameters.append((layer.name, layer.weights[0]))

    candidate_pruning_parameters = sorted(candidate_pruning_parameters, key = lambda x: int(x[0].split("_")[-1]))
    candidate_pruning_parameters = [val for key, val in candidate_pruning_parameters]

    return candidate_pruning_parameters
