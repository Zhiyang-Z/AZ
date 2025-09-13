import jax
import jax.numpy as jnp
import chex
import mctx

from functools import partial

from gomoku import Env, State
from utils import forward
from mcts_func import recurrent_fn, root_fn

# define a class to record selfplay output, each SelfplayOutput instance corresponds to one step in one game,
# Thus, after complete a batch of games, we have selfplay_batch_size * max_num_steps SelfplayOutput and each
# represents a training sample.
@chex.dataclass
class SelfplayOutput:
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray

class SelfPlay:
    def __init__(self, env: Env, simulation: int, max_num_steps: int, search_times: int, num_device: int) -> SelfplayOutput:
        self.env = env
        self.simulation, self.max_num_steps, self.search_times = simulation, max_num_steps, search_times
        self.num_device = num_device
        assert simulation % num_device == 0, "simulation must be divisible by num of device"
        self.sim_per_dev = simulation // num_device
    
    @partial(jax.pmap, static_broadcasted_argnums=0, in_axes=(None, 0, 0))
    def run(self, train_state, rng_key):
        rng_key, subkey = jax.random.split(rng_key, 2)
        env_keys = jax.random.split(subkey, self.sim_per_dev)
        state: State = self.env.reset(env_keys)
        rng_key, subkey = jax.random.split(rng_key, 2)
        step_keys = jax.random.split(subkey, self.max_num_steps)
        def one_step(state: State, rng_key: jnp.ndarray):
            curr_obs = state.observation
            # do the MCTS
            rng_key, subkey = jax.random.split(rng_key, 2)
            policy_output = mctx.gumbel_muzero_policy(
                params=train_state,
                rng_key=subkey,
                root=root_fn(train_state, state),
                recurrent_fn=recurrent_fn,
                num_simulations=self.search_times,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )
            # we get the actions for all the games, then do all the actions.
            new_state, reward, done = self.env.step(state, policy_output.action)
            # discount is for calculate v = reward + discount * v_next,
            # value 0 is for blocking the reward backprop from terminal state when rollout game.
            discount = jnp.where(done, 0, -1).astype(jnp.float32)

            return new_state, SelfplayOutput(
                                                obs=curr_obs,
                                                action_weights=policy_output.action_weights,
                                                reward=reward,
                                                terminated=done,
                                                discount=discount,
                                            )



        _, collected_data = jax.lax.scan(one_step, state, step_keys)

        # collected_data: SelfplayOutput, each field has shape (max_num_steps, sim_per_dev, ...)
        return collected_data

@chex.dataclass
class TrainBatch:
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    # mask: jnp.ndarray

@partial(jax.pmap, in_axes=(None, 0))
def compute_loss_input(sim_per_dev: int, data: SelfplayOutput) -> TrainBatch:
    max_num_step = data.obs.shape[0]
    # # If episode is truncated, there is no value target
    # # So when we compute value loss, we need to mask it
    # value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = max_num_step - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
                                    body_fn,
                                    jnp.zeros(sim_per_dev),
                                    jnp.arange(max_num_step)
                                )
    value_tgt = value_tgt[::-1, :]

    return TrainBatch(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        # mask=value_mask,
    )
