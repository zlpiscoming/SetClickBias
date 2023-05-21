#!/usr/bin/env python
import gym 
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork


def main(robot, task, algo, seed, exp_name, cpu, cus, bid, address, withf, otherwarm, constraint_type):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 6000
    else:
        num_steps = 1e7
        steps_per_epoch = 6000
    num_steps = 2e5
    steps_per_epoch = 4000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    #print(algo)
    #env_name = 'SafetyPointGoal1-v0'
    env_name = 'Hopper-v3'
    #print('before in', withf, otherwarm)
    algo(env_fn=env_name,
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs,
         cus=cus, _bid=bid,
         address=address, withF = withf,
         otherwarm = otherwarm, constraint_type = constraint_type
         )


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == '__main__':
    
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='cpo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--cus', type=str, default='2259')
    parser.add_argument('--bid', type=int, default=287)
    parser.add_argument('--address', type=str, default='1019')
    parser.add_argument('--withf', type=str2bool, default=False)
    parser.add_argument('--otherwarm', type=str2bool, default=False)
    parser.add_argument('--constraint_type', type=str, default='')
    args = parser.parse_args()
    #print(args.withf, args.otherwarm)
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu, args.cus, args.bid, args.address, args.withf, args.otherwarm, args.constraint_type)