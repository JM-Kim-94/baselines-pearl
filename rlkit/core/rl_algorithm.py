import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu

import wandb
from MulticoreTSNE import MulticoreTSNE as TSNE

class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            env_name,
            agent,
            train_tasks,
            eval_tasks,
            indistribution_tasks,
            tsne_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            alpha=1,
            config={},
            tsne_plot_freq=20
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.tsne_plot_freq = tsne_plot_freq
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.indistribution_tasks = indistribution_tasks
        self.tsne_tasks = tsne_tasks

        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.alpha = alpha

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []


        wandb.login(key="7316f79887c82500a01a529518f2af73d5520255")
        wandb.init(
            entity='mlic_academic',
            project='김정모_metaRL_new4',
            group=self.env_name,  
            name='pearl-' + self.env_name + "-rs" + str(config['algo_params']['reward_scale']) + "-alpha" + str(config['algo_params']['alpha']),
            config=config,
        )



    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)

            loss_list = {"q1_pred":[], 
                        "q2_pred":[], 
                        "v_pred":[], 
                        "target_v_values":[], 
                        "kl_loss":[], 
                        "q_target":[], 
                        "qf_loss":[], 
                        "v_target":[], 
                        "policy_loss":[], 
                        "log_pi":[]
                        }
            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                
                loss_dict = self._do_training(indices)
                for k in loss_dict.keys():
                    loss_list[k].append(loss_dict[k])

                self._n_train_steps_total += 1

            for k in loss_list.keys():
                loss_list[k] = np.mean(loss_list[k])

            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_, loss_list)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch, loss_list):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch, loss_list)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run, return_z=False):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        task_z = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)
                task_z.append(self.agent.z)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        if return_z:
            return paths, task_z[0]
        else:
            return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def _do_tsne_plot(self, indices, epoch):
        # collect_paths 활용
        # asdf asdfas
        pass

    def evaluate(self, epoch, loss_list):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        ### Indistribution 
        if self.env_name == "ant-dir-4개":
            indistribution_final_returns = [0]
        else:
            eval_util.dprint('evaluating on {} test tasks'.format(len(self.indistribution_tasks)))
            indistribution_final_returns, indistribution_online_returns = self._do_eval(self.indistribution_tasks, epoch)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)


        # if epoch % self.tsne_plot_freq == 0:
        #     print("TSNE plot 시작")
        #     # self._do_tsne_eval(self.tsne_tasks, epoch)
        #     self._do_tsne_plot(self.tsne_tasks, epoch)
        #     print("TSNE plot 끝")

        

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_indistribution_return = np.mean(indistribution_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))


        wandb_log_dict = {
            "Eval/train_avg_return": avg_train_return,
            "Eval/test_avg_return": avg_test_return,
            "Eval/indistribution_avg_return": avg_indistribution_return,
        }
        wandb_log_dict.update(loss_list)        

        wandb.log(wandb_log_dict, step=self._n_env_steps_total)




        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

