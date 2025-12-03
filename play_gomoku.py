from gomoku import Env, State
import jax
import jax.numpy as jnp
from network import AZNet
import flax
from flax.training import train_state
import optax
from typing import Any
import orbax
from utils import forward
import orbax.checkpoint
import mctx
from common import random_recurrent_fn, random_root_fn
import os
from time import sleep

# initialize the environment
env = Env("gomoku-15x15")
# initialize the board
rngkey, subkey = jax.random.split(jax.random.PRNGKey(0), 2)
env_keys = jax.random.split(subkey, 1)
state = env.reset(env_keys)

# define the netwok
# load the opponent model
az_net_opp = AZNet(num_actions=225)  # 4672 is the number of possible moves in chess
# Initialize network, warm-up.
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
restore_path = "/home/zhiyang/projects/class_projects/MARL/final/AZ_gomoku_000984"  # path to the opponent model
variables = orbax_checkpointer.restore(restore_path)
model_state_opp, params_opp = variables['model_state'], variables['params']
class TrainState(train_state.TrainState):
    model_state: Any
    metrics: dict
train_state = TrainState.create(
    apply_fn=az_net_opp.apply,
    params=params_opp,
    model_state=model_state_opp,
    tx=optax.adamw(0.001),
    metrics={},
)

while state.done[0] == False:
    # os.system('cls' if os.name == 'nt' else 'clear')
    env.print_board(state)
    action = input("Your move: ")
    action = tuple(map(int, action.split()))
    action = action[0] * 15 + action[1]
    state, reward, done = env.step(state, jnp.array([action], dtype=jnp.uint8))
    if done:
        print('You win!')
        break
    env.print_board(state)
    print('machine is thinking...', end="")
    sleep(1.5)
    # policy_output = mctx.gumbel_muzero_policy(
    #                                                 params=None,
    #                                                 rng_key=subkey,
    #                                                 root=random_root_fn(state),
    #                                                 recurrent_fn=random_recurrent_fn,
    #                                                 num_simulations=128,
    #                                                 invalid_actions=~state.legal_action_mask,
    #                                                 qtransform=mctx.qtransform_completed_by_mix_value,
    #                                                 gumbel_scale=1.0,
    #                                             )
    # computer_logits = jnp.log(policy_output.action_weights)
    # # mask invalid actions
    # computer_logits = jnp.where(state.legal_action_mask, computer_logits, jnp.finfo(computer_logits.dtype).min)
    # computer_action = jnp.argmax(computer_logits, axis=-1)
    # jax.debug.print("computer action: {a}", a=computer_action)

    (computer_logits, _) = forward(train_state, state.observation)
    # mask invalid actions
    computer_logits = jnp.where(state.legal_action_mask, computer_logits, jnp.finfo(computer_logits.dtype).min)
    computer_action = jnp.argmax(computer_logits, axis=-1)
    print(f"put({computer_action//15},{computer_action%15})")

    state, reward, done = env.step(state, computer_action)
    if done:
        print('machine win!')
        break

