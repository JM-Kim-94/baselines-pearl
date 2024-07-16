"""MuJoCo150버전"""
import numpy as np
import random
# from rlkit.envs.ant import AntEnv
from gym.envs.mujoco import AntEnv as AntEnv

from . import register_env


# "n_train_tasks": 150,
# "n_eval_tasks": 4,
# "n_indistribution_tasks": 8,
# "eval_tasks_list": [[1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75]],
# "indistribution_tasks_list": [[0.5,  0], [0, 0.5 ], [-0.5,  0], [0, -0.5 ],
#                                 [2.75, 0], [0, 2.75], [-2.75, 0], [0, -2.75]]

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal-inter')
class AntGoalInterEnv(AntEnv):
    def __init__(self, n_train_tasks, n_eval_tasks, n_indistribution_tasks, eval_tasks_list, indistribution_tasks_list):  # ood = "inter" or "extra"
        self._task = {}
        self.tasks = self.sample_tasks(n_train_tasks, n_eval_tasks, eval_tasks_list, indistribution_tasks_list)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']

        super(AntGoalInterEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def sample_tasks(self, n_train_tasks, n_eval_tasks, eval_tasks_list, indistribution_tasks_list):

        goal_train = []
        for i in range(n_train_tasks):
            prob = random.random()  # np.random.uniform()
            if prob < 4.0 / 15.0:
                r = random.random() ** 0.5  # [0, 1]
            else:
                # r = random.random() * 0.5 + 2.5  # [2.5, 3.0]
                r = (random.random() * 2.75 + 6.25) ** 0.5
            theta = random.random() * 2 * np.pi  # [0.0, 2pi]
            goal_train.append([r * np.cos(theta), r * np.sin(theta)])      

        goal_test = eval_tasks_list
        goal_indistribution = indistribution_tasks_list


        """tsne-tasks"""
        theta_list = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * np.pi / 4
        train_r_list = np.array([0.5, 1.0, 2.5, 3.0])
        test_r_list = np.array([1.5, 2.0])
        train_tsne_tasks_list, test_tsne_tasks_list = [], []

        for r in train_r_list:
            for theta in theta_list:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                train_tsne_tasks_list.append([x, y])

        for r in test_r_list:
            for theta in theta_list:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                test_tsne_tasks_list.append([x, y])

        goal_tsne = train_tsne_tasks_list + test_tsne_tasks_list

        goals = goal_train + goal_test + goal_indistribution + goal_tsne
        goals = np.array(goals)

        tasks = [{'goal': goal} for goal in goals]

        return tasks

    def _get_obs(self):
        o = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        return o

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()





