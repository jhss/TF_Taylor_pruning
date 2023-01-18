# (1) 기존에 pre-trained network에 gate layer를 추가한다.
# (2) gate layer에 연결된 layer를 기준으로 pruning score를 계산한다.
# (3) 일정 score 이하인 gate를 0으로 설정하면, 해당 gate에 연결된 뉴런이 반영되지 않는다.
# method number: 22
class tensorflow_pruning(object):
    def __init__(self, model, parameters, gradients):
        # Complete Implementation of basic structure '__init__' [23.01.18]
        self.model = model
        self.gradients = gradients
        self.parameters = parameters

        # layer마다 neuron 별로 pruning score 계산해서 저장
        self.pruned_neurons = 0  # store the number of pruned neurons
        self.prune_per_iteration = 100

        self.prune_network_criteria = list()
        self.prune_network_accumulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}
        self.pruning_gates = list()
        # unneccesary self.prune_layers due to True for all layers

        # [2023.01.17] cur_layer_criteria를 tensorflow 기준으로 변경해야함
        # self.model.layers.get_weights()에서 parameter 크기로 설정하려고 하는데
        # PyTorh shape랑 맞춰줘야 해서 어떻게 설정해야할지 고민하던 중에 그만뒀음
        for parameter in self.parameters:
            n_units = len(parameter)
            cur_layer_criteria = [0.0 for unit in range(n_units)]
            self.prune_network_criteria.append(cur_layer_criteria)

            # 각 key별로 layer 개수만큼 list 추가
            for key in self.prune_network_accumulate.keys():
                self.prune_network_accumulate[key].append(list())

            # layer마다 뉴런 개수만큼 gate 추가 (값이 0이면 pruned 상태)
            self.pruning_gates.append(np.ones(n_units,))

    def do_step(self, loss):
        '''
        지난 iteration 때 적용한 pruning으로 인해 loss가 많이 변했으면 pruning을 적용하지 않음

        변화가 threshold보다 작으면 pruning 적용

        '''

        # enforce_pruning

        # add_criteria
        self.add_criteria()
        # util_add_loss: loss tracker (not important)

        # report_loss_neuron : logging (not important)

        # compute_saliency
        self.compute_saliency()
        # set_momentum_zero_sgd
        #

    def enforce_pruning(self):
        pass

    # [Stage]
    def add_criteria(self):

        for idx, (paremeter, gradient) in enumerate(zip(self.parameters, self.gradients)):

            criteria_per_layer = tf.pow(tf.multiply(parameter, gradient), 2)

            mult = 3.0
            if idx == 1: mult = 4.0
            elif idx == 2: mult = 6.0

            criteria_per_layer /= mult

            if self.cur_iterations == 0:
                self.prune_network_accumulate['by_layer'][layer] = criteria_for_layer
            else:
                self.prune_network_accumulate['by_layer'][layer] += criteria_for_layer

        self.cur_iterations += 1


    # list_criteria_per_layer에 있는 layer를 하나의 group으로 묶는다.
    # 하나의 layer에 있는 모든 뉴런의 score를 더해서 layer마다 저장한다.
    # [ [(n1, score), (n2, score), ... ], [ ]] -> [[layer1, score], [layer2, score], ... ]
    def group_criteria(list_criteria_per_layer, group_size = 1):

        groups = list()

        for layer in list_criteria_per_layer:
            layer_groups = list()
            indices = np.argsort(layer)
            for group_id in range(int(np.ceil(len(layer)/group_size))):
                current_group = slice(group_id*group_size, min((group_id+1)*group_size, len(layer)))
                values = [layer[ind] for ind in indeces[current_group]]
                group = [indeces[current_group], sum(values)]

                layer_groups.append(group)
            groups.append(layer_groups)

        return groups

    def compute_saliency(self):


        if validation_error > self.pruning_threshold:
            print(f"Skip Pruning valid error {validation_error:.4f} > threshold: {self.pruning_threshold:.4f}")
            return -1

        if self.max_pruning_iterations <= self.cur_pruning_iterations:
            return -1


        for layer, parameter in enumerate(self.parameters):

            contribution = self.prune_network_accumulate["by_layer"][layer] / self.cur_iterations
            # use momentum to accumulate criteria over several pruning iterations.
            # weighted average between previous average and current contribution.
            # 각 layer마다 계산된 criteria를 할당한다.
            self.prune_network_accumulate["averaged"][layer] = contribution
            current_layer = self.prune_network_accumulate["averaged"][layer]

            # l2_normalization_per_layer = False, skip
            self.prune_network_accumulate["averaged_cpu"][layer] = current_layer

            # [CHECK] expected unit: 1 for all layer
            ''' pruning이 이미 되었으면 contribution을 0으로 설정한다. '''
            for unit in range(len(parameter)):
                criterion_now = current_layer[unit]
                self.prune_network_criteria[layer][unit] =  criterion_now * self.pruning_gates[layer][unit]

        '''
        각 뉴런마다 할당한 contribution을 기반으로, group의 contribution을 계산한다.

        [CHECK] group criteria가 어떻게 되는지 확인해봐야함
        '''
        # create groups per layer
        groups = self.group_criteria(self.prune_network_criteria, group_size=self.group_size)

         # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)//self.group_size - 1

        #if self.prune_neurons_max != -1:
        #    prune_neurons_now = min(len(all_criteria)-1, min(prune_neurons_now, self.prune_neurons_max//self.group_size - 1))

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]


        for layer, parameter in enumerate(self.parameters):

            '''
            그룹안에 있는 뉴런의 기여도가 threshold보다 작을경우, 해당 그룹 뉴런을 모드 제거한다.

            gate는 해당 layer 사용유무를 나타내고
            parameter는 실제 모델에 변화를 기록하는것
            '''

            cur_layer_weight = self.model.layers[layer].get_weights()[0]

            for group in groups[layer]:
                if group[1] <= threshold_now:
                    for unit in group[0]:
                        # do actual pruning
                        self.pruning_gates[layer][unit] *= 0.0
                        # [NOTE] 특정 위치에 0을 대입해야함 [Tensorflow]
                        cur_layer_weight[unit] = 0.0

                        #self.parameters[layer].data[unit] *= 0.0
            self.model.layers[layer].set_weights(cur_layer_weight)

        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.pruned_neurons = all_neuron_units - neuron_units

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

            all_neuron_units += len( parameter )
            for unit in range(len( parameter )):
                if len(parameter.size()) > 1:
                    statistics = parameter.abs().sum()
                else:
                    statistics = parameter[unit]

                if statistics > 0.0:
                    neuron_units += 1

        return all_neuron_units, neuron_units

# [CHECK] 개수 36인지 Check
def select_pruning_parameters(model):

    candidate_pruning_parameters = []

    for layer in model.layers:
        if 'gate' in layer.name:
            candidate_pruning_parameters.append(layer.get_weights()[0])

    return candidate_pruning_parameters
