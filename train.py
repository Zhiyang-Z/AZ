import datetime
import os
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import optax
import wandb

import flax
from flax.training import train_state
import optax
from flax.jax_utils import unreplicate
import orbax.checkpoint

from typing import Any

from gomoku import Env, State
from network import AZNet
from utils import train_step, forward
from common import config

import os
os.environ["WANDB_MODE"] = "disabled"

# Initialize environment
from common import env

# define a class to record selfplay output, each SelfplayOutput instance corresponds to one step in one game,
# Thus, after complete a batch of games, we have selfplay_batch_size * max_num_steps SelfplayOutput and each
# represents a training sample.
from selfplay import SelfPlay, SelfplayOutput, compute_loss_input, TrainBatch
selfplay = SelfPlay(env, config.selfplay_batch_size, config.max_num_steps, config.num_search, jax.local_device_count())

from common import evaluate_with_random

if __name__ == "__main__":
    wandb.init(project="chess-az", config=config.model_dump())
    # define neural network
    az_net = AZNet(num_actions=19*19)
    # Initialize network, warm-up.
    rng_key, subkey = jax.random.split(jax.random.PRNGKey(config.seed), 2)
    dummy_input = jnp.zeros((32, 19, 19, 17))
    model_variables = az_net.init(subkey, dummy_input, is_training=True)
    model_state, params = flax.core.pop(model_variables, "params")
    # define train state
    class TrainState(train_state.TrainState):
        model_state: Any
        metrics: dict
    train_state = TrainState.create(
        apply_fn=az_net.apply,
        params=params,
        model_state=model_state,
        tx=optax.adamw(config.learning_rate, weight_decay=1e-4),
        metrics={},
    )
    print(az_net.tabulate(subkey, dummy_input, is_training=True))
    # detect devices
    devices = jax.local_devices()
    num_devices = len(devices)
    print(f"Found {num_devices} devices: {devices}, putting model on devices.")
    # put model into all devices for parallel.
    train_state = jax.device_put_replicated(train_state, devices)

    # main training stage
    iter, hours, frames = 0, 0.0, 0
    log = {"iteration": iter, "hours": hours, "frames": frames}
    print(log)
    wandb.log(log)
    print('training start at ', datetime.datetime.now())
    while True:
        if iter % config.eval_interval == 0:
            # evaluate the model
            # # load the opponent model
            # az_net_opp = AZNet(num_actions=4672)  # 4672 is the number of possible moves in chess
            # # Initialize network, warm-up.
            # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            # restore_path = "/home/zhiyang/projects/class_projects/MARL/pgx/examples/chess/AZ_chess_000040"  # path to the opponent model
            # variables = orbax_checkpointer.restore(restore_path)
            # model_state_opp, params_opp = variables['model_state'], variables['params']
            # train_state_opp = TrainState.create(
            #     apply_fn=az_net_opp.apply,
            #     params=params_opp,
            #     model_state=model_state_opp,
            #     tx=optax.adamw(config.learning_rate),
            #     metrics={},
            # )
            # train_state_opp = jax.device_put_replicated(train_state_opp, devices)
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            # R = evaluate(keys, train_state, train_state_opp)
            R = evaluate_with_random(keys, train_state)
            log.update(
                {
                    f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )
            print(log)

            # store checkpoints
            save_path = f'/home/zhiyang/projects/class_projects/MARL/final/saved_params/AZ_gomoku_{iter:06d}'
            # unreplicate the whole TrainState
            train_state_to_save = unreplicate(train_state)
            ckpt = {
                'params': train_state_to_save.params,
                'model_state': train_state_to_save.model_state,
                'opt_state': train_state_to_save.opt_state
                }
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            orbax_checkpointer.save(save_path, ckpt)
            del train_state_to_save # relaese manually for long run.

        start_time = time.time()
        # run selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data = selfplay.run(train_state, keys) # data shape: (num_devices, max_num_steps, selfplay_batch_size // num_devices, ...)
        # Now, we have collected a batch of data, but data['reward'] is the immediate reward.
        # We need to compute the n-step return for each step.
        samples = compute_loss_input(config.selfplay_batch_size // num_devices, data)
        # shuffle and batch the samples for training
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )
        # train the model
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            train_state, train_info = train_step(train_state, minibatch)
            policy_losses.append(train_info['policy_loss'].mean().item())
            value_losses.append(train_info['value_loss'].mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        end_time = time.time()
        iter += 1
        hours += (end_time - start_time) / 3600.0
        # print and log the training info
        log = {
                "iteration": iter,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        print(log)
        wandb.log(log)


