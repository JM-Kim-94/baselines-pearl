import numpy as np
from rand_param_envs.walker_mass_inter import WalkerMassInterEnv

from . import register_env


# "env_params": {
# "n_train_tasks": 100,
# "n_eval_tasks": 5,
# "n_indistribution_tasks": 4,
# "eval_tasks_list": [0.75, 1.25, 1.75, 2.25, 2.75],
# "indistribution_tasks_list": [0.1, 0.25, 3.1, 3.25]
# }

@register_env('walker-mass-inter')
class WalkerMassInterWrappedEnv(WalkerMassInterEnv):
    def __init__(self, n_train_tasks, n_eval_tasks, n_indistribution_tasks, eval_tasks_list, indistribution_tasks_list):

        super(WalkerMassInterWrappedEnv, self).__init__()
        
        self.tasks, self.tasks_value = self.sample_tasks(n_train_tasks, eval_tasks_list, indistribution_tasks_list)
        
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
