
class tensorflow_pruning(object):
    def __init__(self, ):



    def do_step(self, loss):
        '''
        지난 iteration 때 적용한 pruning으로 인해 loss가 많이 변했으면 pruning을 적용하지 않음

        변화가 threshold보다 작으면 pruning 적용

        '''

        # enforce_pruning

        # add_criteria

        # util_add_loss

        # report_loss_neuron : logging

        # compute_saliency

        # set_momentum_zero_sgd
        #

    def group_criteria(list_criteria_per_layer, group_size = 1):

        groups = list()

    def compute_saliency(self, ):


        if validation_error > self.pruning_threshold:
            print(f"Skip Pruning valid error {validation_error:.4f} > threshold: {self.pruning_threshold:.4f}")

            return -1

        if self.max_pruning_iterations <= self.cur_pruning_iterations:

            return -1


        for layer, apply_prune in enumerate(self.pruned_layers):

            if not apply_prune:
                continue

            contribution = self.prune_network_accomulate["by_layer"][layer] / self.iterations_done
            self.prune_network_accomulate["averaged"][layer] = contribution
            current_layer = self.prune_network_accomulate["averaged"][layer]
            self.prune_network_accomulate["averaged_cpu"][layer] = current_layer

            ''' pruning이 이미 되었으면 contribution을 0으로 설정한다. '''
            for unit in range(len(self.parameters[layer])):
                criterion_now = current_layer[unit]
                self.prune_network_criteria[layer][unit] =  criterion_now * self.pruning_gates[layer][unit]

        '''
        각 뉴런마다 할당한 contribution을 기반으로, group의 contribution을 계산한다.

        group criteria가 어떻게 되는지 확인해봐야함
        '''
        # create groups per layer
        groups = self.group_criteria(self.prune_network_criteria, group_size=self.group_size)

         # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)//self.group_size - 1

        if self.prune_neurons_max != -1:
            prune_neurons_now = min(len(all_criteria)-1, min(prune_neurons_now, self.prune_neurons_max//self.group_size - 1))

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]


        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if self.prune_per_iteration == 0:
                continue
            '''
            그룹안에 있는 뉴런의 기여도가 threshold보다 작을경우, 해당 그룹 뉴런을 모드 제거한다.

            근데 gate와 parameter를 둘다 왜 0으로 하는지 모르겠음.
            '''
            for group in groups[layer]:
                if group[1] <= threshold_now:
                    for unit in group[0]:
                        # do actual pruning
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0
