from pydantic import BaseModel
from gomoku import Env
import jax
import jax.numpy as jnp
import mctx

from utils import forward
from omegaconf import OmegaConf

# Initialize environment
env = Env("gomoku-15x15")

# In alphazero, there is little difference between standard MCTS algorithm.
# In standard MCTS, we have 4 steps: selection, expansion, simulation and backpropogation.
# In alphazero, we replace the simulation step with a neural network, which directly outputs the value estimation.
# Here, change the terminology in alphzero, we call selection + expansion as one simulation step.
# Then, here is only 3 steps: simulation, expansion (with NN evaluation) and backpropogation in alphzero.
class Config(BaseModel):
    env_id: str = "gomoku-19x19"
    seed: int = 0 # For chess game, seed doesn't matter, we always start from empty board
    # network params
    num_filters: int = 256
    num_residual_blocks: int = 16
    # selfplay params
    selfplay_batch_size: int = 1024 # This is also the total number of games, train once after one batch of games complete.
    num_search: int = 32 # number of simulations per move (one simulation is a loop of simulation, expansion (with NN evaluation) and backpropogation).
    max_num_steps: int = 15*15 # each game can have at most 256 steps to play.
    # training params
    training_batch_size: int = 512 # after MCTS, we have selfplay_batch_size * max_num_steps samples to train on.
    learning_rate: float = 0.0001
    # eval params
    eval_interval: int = 8

    class Config:
        extra = "forbid"
conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

def random_root_fn(state):
    logits = jnp.zeros_like(state.legal_action_mask, dtype=jnp.float32)
    # mask invalid actions
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return mctx.RootFnOutput(prior_logits=logits, value=jnp.full(state.current_player.shape, 0.5), embedding=state)

def random_recurrent_fn(train_state, rng_key: jnp.ndarray, action: jnp.ndarray, state):
    del rng_key # board game does not need stochasticity in the env.

    current_player = state.current_player
    state, reward, done = env.step(state, action)

    logits = jnp.zeros_like(state.legal_action_mask, dtype=jnp.float32)
    # mask invalid actions
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    # discount=-1 to flip the player, discount=0 to block the reward backprop from terminal state on MC tree.
    discount = jnp.where(done, 0, -1).astype(jnp.float32)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=jnp.full(state.current_player.shape, 0.5),
    )
    return recurrent_fn_output, state

@jax.pmap
def evaluate_with_random(rng_key, train_state):
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 1
    # detect devices
    devices = jax.local_devices()
    num_devices = len(devices)

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = env.reset(keys)

    def body_fn(val):
        key, state, R = val
        key, subkey = jax.random.split(key)
        (my_logits, _) = forward(train_state, state.observation)
        # mask invalid actions
        my_logits = jnp.where(state.legal_action_mask, my_logits, jnp.finfo(my_logits.dtype).min)
        policy_output = mctx.gumbel_muzero_policy(
                                                    params=None,
                                                    rng_key=subkey,
                                                    root=random_root_fn(state),
                                                    recurrent_fn=random_recurrent_fn,
                                                    num_simulations=config.num_search,
                                                    invalid_actions=~state.legal_action_mask,
                                                    qtransform=mctx.qtransform_completed_by_mix_value,
                                                    gumbel_scale=1.0,
                                                )
        opp_logits = jnp.log(policy_output.action_weights)
        # mask invalid actions
        opp_logits = jnp.where(state.legal_action_mask, opp_logits, jnp.finfo(opp_logits.dtype).min)

        is_my_turn = (state.current_player == my_player)
        logits = jnp.where(is_my_turn.reshape((-1, 1)), my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state, reward, done = env.step(state, action)
        R = R + jnp.where(is_my_turn, reward, -reward)
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].done.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R
