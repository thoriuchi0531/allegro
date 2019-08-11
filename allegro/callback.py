from operator import gt, lt
import numpy as np


class ReduceLrOnPlateau:
    def __init__(self, monitor, factor=0.1, patience=10, min_lr=0,
                 cooldown=0):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.is_min_lr = False
        self.cooldown = cooldown
        self.cooldown_counter = 0

        self.best_score = None
        self.best_iter = None
        self.cmp_op = None
        self.monitored_score_idx = None

        self.order = 99  # callback priority

    def _init(self, env):
        """ Initialise parameters based on the given callback environment """
        print(f'Learning rate is reduced by {self.factor} if validation scores '
              f'do not improve for {self.patience} rounds.')

        if not env.evaluation_result_list:
            raise ValueError('For reduce lr on plateau, '
                             'at least one dataset and eval metric is required for evaluation')
        for idx, i in enumerate(env.evaluation_result_list):
            if i[1] == self.monitor:
                self.monitored_score_idx = idx
                break
        if self.monitored_score_idx is None:
            raise ValueError(f'{self.monitor} is not found in evaluation results')

        self.best_iter = 0
        if env.evaluation_result_list[self.monitored_score_idx][3]:
            # larger is better
            self.best_score = -np.Inf
            self.cmp_op = gt
        else:
            self.best_score = np.Inf
            self.cmp_op = lt

    def __call__(self, env):
        if self.cmp_op is None:
            self._init(env)

        if self.in_cooldown():
            self.cooldown_counter -= 1

        if not self.is_min_lr:
            eval_ret = env.evaluation_result_list[self.monitored_score_idx]
            score = eval_ret[2]
            # Only consider reducing the learning rate when it hasn't reached
            # the minimum yet
            if self.cmp_op(score, self.best_score):
                # Score improved.
                self.best_score = score
                self.best_iter = env.iteration
            elif ((not self.in_cooldown()) and
                  (env.iteration - self.best_iter >= self.patience)):
                # when `cooldown_counter == 0`
                old_lr = env.params['learning_rate']
                new_lr = max(self.min_lr, old_lr * self.factor)
                print(f'[{env.iteration + 1}]\tLearning rate reduced to {new_lr}')
                if new_lr == self.min_lr:
                    self.is_min_lr = True

                lr_dict = {'learning_rate': new_lr}
                env.model.reset_parameter(lr_dict)
                env.params.update(lr_dict)
                self.cooldown_counter = self.cooldown

    def in_cooldown(self):
        return self.cooldown_counter > 0
