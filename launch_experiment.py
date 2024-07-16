"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


def experiment(variant):

    print("variant", variant)

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(  ENVS[variant['env_name']]  (**variant['env_params'])  )

    tasks, total_tasks_dict_list = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )

    num_train = variant["n_train_tasks"]  # 50
    num_eval = variant["n_eval_tasks"]  # 5
    num_indistribution = variant["n_indistribution_tasks"]  # 5
    num_tsne = variant["n_tsne_tasks"]  # 5
    print("num_train", num_train)
    print("num_test", num_eval)
    print("num_indistribution", num_indistribution)
    print("num_tsne", num_tsne)

    print("train_tasks = ", tasks[: num_train])  # 0~49
    print("eval_tasks = ", tasks[num_train:  num_train + num_eval])  # 50 ~ 54
    print("indistribution_tasks = ", tasks[num_train + num_eval: num_train + num_eval + num_indistribution])  # 55 ~ 59
    print("tsne_tasks = ", tasks[num_train + num_eval + num_indistribution: num_train + num_eval + num_indistribution + num_tsne])

    train_tasks = tasks[: num_train]
    eval_tasks = tasks[num_train:  num_train + num_eval]
    indistribution_tasks = tasks[num_train + num_eval: num_train + num_eval + num_indistribution]
    tsne_tasks = tasks[num_train + num_eval + num_indistribution: num_train + num_eval + num_indistribution + num_tsne]

    """ tsne 그리는 태스크들 
    ant-dir-2 : [0.0, 1 * np.pi / 4, 0.5 * np.pi, 3 * np.pi / 4, 7 * np.pi / 4]
    ant-dir-4 : [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]
    ant-goal-inter : [[0.5,  0], [0, 0.5 ], [-0.5,  0], [0, -0.5 ],
                      [1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75],
                      [2.75, 0], [0, 2.75], [-2.75, 0], [0, -2.75]]
    cheetah-vel-inter, walker-mass-inter, hopper-mass-inter : [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25]
    """

    algorithm = PEARLSoftActorCritic(
        env=env,
        env_name=variant['env_name'],
        train_tasks=train_tasks, 
        eval_tasks=eval_tasks, 
        indistribution_tasks=indistribution_tasks,
        tsne_tasks=tsne_tasks,
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        config=variant,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--reward_scale', default=5.0, type=float)
@click.option('--alpha', default=1.0, type=float)
def main(config, gpu, docker, debug, reward_scale, alpha):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    variant['algo_params']['reward_scale'] = reward_scale
    variant['algo_params']['alpha'] = alpha

    experiment(variant)

if __name__ == "__main__":
    main()

