import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from functools import partial

@partial(jax.pmap, axis_name="i") # axis_name for using collective operations.
def train_step(train_state: train_state.TrainState, batch):
    def loss_fn(params):
        (logits, v), new_model_state = train_state.apply_fn({'params': params, **train_state.model_state},
                                                        batch.obs,
                                                        mutable=list(train_state.model_state.keys()),
                                                        is_training=True)
        policy_loss = optax.softmax_cross_entropy(logits, batch.policy_tgt).mean()
        value_loss = (optax.l2_loss(v, batch.value_tgt) * batch.mask).mean()
        loss = policy_loss + value_loss
        return loss, (policy_loss, value_loss, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss, new_model_state)), grads = grad_fn(train_state.params)
    grads = jax.lax.pmean(grads, axis_name="i")
    train_state = train_state.apply_gradients(grads=grads, model_state=new_model_state)
    
    train_info = {
        'loss': loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
    }
    return train_state, train_info


def forward(train_state: train_state.TrainState, x):
    logits, v = train_state.apply_fn({'params': train_state.params, **train_state.model_state},
                                         x,
                                         mutable=False,
                                         is_training=False)
    return logits, v

# @jax.pmap
# def evaluate(rng_key, train_state, train_state_opp):
#     """A simplified evaluation by sampling. Only for debugging. 
#     Please use MCTS and run tournaments for serious evaluation."""
#     my_player = 0
#     # detect devices
#     devices = jax.local_devices()
#     num_devices = len(devices)

#     key, subkey = jax.random.split(rng_key)
#     batch_size = config.selfplay_batch_size // num_devices
#     keys = jax.random.split(subkey, batch_size)
#     state = jax.vmap(env.init)(keys)

#     def body_fn(val):
#         key, state, R = val
#         (my_logits, _) = forward(train_state, state.observation)
#         opp_logits, _ = forward(train_state_opp, state.observation)
#         is_my_turn = (state.current_player == my_player).reshape((-1, 1))
#         logits = jnp.where(is_my_turn, my_logits, opp_logits)
#         key, subkey = jax.random.split(key)
#         action = jax.random.categorical(subkey, logits, axis=-1)
#         state = jax.vmap(env.step)(state, action)
#         R = R + state.rewards[jnp.arange(batch_size), my_player]
#         return (key, state, R)

#     _, _, R = jax.lax.while_loop(
#         lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
#     )
#     return R