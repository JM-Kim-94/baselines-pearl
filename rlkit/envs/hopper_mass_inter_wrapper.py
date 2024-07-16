


import numpy as np
from rand_param_envs.hopper_mass_inter import HopperMassInterEnv

from . import register_env


@register_env('hopper-mass-inter')
class HopperMassInterWrappedEnv(HopperMassInterEnv):

    def __init__(self, n_train_tasks, n_eval_tasks, n_indistribution_tasks, eval_tasks_list, indistribution_tasks_list, tsne_tasks_list):

        super(HopperMassInterWrappedEnv, self).__init__()
        
        self.tasks, self.tasks_value = self.sample_tasks(n_train_tasks, eval_tasks_list, indistribution_tasks_list, tsne_tasks_list)
        
        self.reset_task(0)
    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks_value

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
