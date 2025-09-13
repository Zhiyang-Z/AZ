import jax
import jax.numpy as jnp
import mctx

from gomoku import Env, State
from utils import forward

# Initialize environment
from common import env

# To use mctx library, we need to define 2 functions: root_fn and recurrent_fn.
def root_fn(train_state, state):
    # get the (logits, value) from the network
    (logits, value) = forward(train_state, state.observation)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True) # for numerical stability
    # mask invalid actions
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

def recurrent_fn(train_state, rng_key: jnp.ndarray, action: jnp.ndarray, state: State):
    del rng_key # board game does not need stochasticity in the env.

    current_player = state.current_player
    state, reward, done = env.step(state, action)

    (logits, value) = forward(train_state, state.observation)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True) # for numerical stability
    # mask invalid actions
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    value = jnp.where(state.done, 0.0, value)
    # discount=-1 to flip the player, discount=0 to block the reward backprop from terminal state on MC tree.
    discount = jnp.where(done, 0, -1).astype(jnp.float32)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state
