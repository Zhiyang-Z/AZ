import jax
import jax.numpy as jnp
import chex
from jax.experimental import checkify

from functools import partial

@chex.dataclass
class State:
    # Observation(a binary tensor): stack 8 history and current player plane
    # We follow the stack way in https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    # Current player's and opponent's pieces alternates [X,Y,X,Y,...,X,Y,P] and append a player plane
    # in tail. In player plane: 1 for 'o' and 0 for 'x'. So 2*8+1=17 channels in total.
    observation: jnp.ndarray
    current_player: jnp.ndarray # 1 or 0 and begin with 1, then alternate
    legal_action_mask: jnp.ndarray # 1 for legal action and 0 for illegal action, size: board_size^2
    done: jnp.ndarray # True or False

class Env:
    def __init__(self, name: str):
        if name == "gomoku-19x19":
            self._board_size = 19
        elif name == "gomoku-15x15":
            self._board_size = 15
        elif name == "gomoku-9x9":
            self._board_size = 9
        else:
            raise ValueError(f"Unknown environment name: {name}")
        self._action_size = self._board_size * self._board_size
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    @partial(jax.vmap, in_axes=(None, 0))
    def reset(self, rngkey) -> State:
        # rngkey here is not used, just for vmap. Board game always start from the same state.
        board_history=jnp.zeros((self._board_size, self._board_size, 16), dtype=jnp.float32)
        first_player_plane = jnp.ones((self._board_size, self._board_size, 1), dtype=jnp.float32) # 'o' plays first
        init_observation = jnp.concatenate([board_history, first_player_plane], axis=-1)
        chex.assert_shape(init_observation, (self._board_size, self._board_size, 17))
        state = State(
                observation=init_observation,
                current_player=jnp.array(1, dtype=jnp.int8),
                legal_action_mask=jnp.ones((self._action_size,), dtype=jnp.int8),
                done=jnp.array(False, dtype=bool)
            )
        return state
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    def _horizontals(self, board: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([
            board[i, j:j+5]
            for i in range(board.shape[0])
            for j in range(board.shape[1] - 4)
        ])
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    def _verticals(self, board: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([
            board[i:i+5, j]
            for i in range(board.shape[0] - 4)
            for j in range(board.shape[1])
        ])
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    def _diagonals(self, board: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([
            jnp.diag(board[i:i+5, j:j+5])
            for i in range(board.shape[0] - 4)
            for j in range(board.shape[1] - 4)
        ])
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    def _antidiagonals(self, board: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([
            jnp.diag(board[i:i+5, j:j+5][::-1])
            for i in range(board.shape[0] - 4)
            for j in range(board.shape[1] - 4)
        ])
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    def _check_win(self, board: jnp.ndarray) -> bool:
        chex.assert_shape(board, (self._board_size, self._board_size))
        seg_h_v = jnp.concatenate([self._horizontals(board), self._verticals(board)])
        seg_d_a = jnp.concatenate([self._diagonals(board), self._antidiagonals(board)])
        win = jnp.any(jnp.all(seg_h_v == 1, axis=1)) | jnp.any(jnp.all(seg_d_a == 1, axis=1))
        return win
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    @partial(jax.vmap, in_axes=(None,0,0))
    def step(self, state: State, action: jnp.ndarray) -> tuple:
        # print('compile...') # jax.debug.print("Running step on {x}", x=action.device())
        opponent_piece, player_piece = state.observation[..., 0], state.observation[..., 1]
        # Action should be legal.(another way is to learn legal move itself with reward panalty.)
        # chex.assert_equal(state.legal_action_mask[action], jnp.array(1, dtype=jnp.int8), 'Illegal action!')
        row, col = action // self._board_size, action % self._board_size
        new_piece_plane = player_piece.at[row, col].set(1) # expect for new memory allocated tensor (JAX immutable)
        # check win
        current_board = jnp.logical_or(opponent_piece, new_piece_plane) # combine opponent's and player's pieces to get current board.
        win = self._check_win(new_piece_plane)
        # assemble returns
        next_player_plane = jnp.full((self._board_size, self._board_size, 1), 1-state.current_player, dtype=jnp.float32) # flip current player
        new_observation = jnp.concatenate([new_piece_plane[..., jnp.newaxis], state.observation[..., :15], next_player_plane], axis=-1)
        chex.assert_shape(new_observation, (self._board_size, self._board_size, 17))
        legal_action_mask = jnp.reshape(current_board == 0, (-1,)).astype(jnp.int8)
        # calculate reward and check done
        reward = jnp.where(win & jnp.logical_not(state.done), 1, 0).astype(jnp.int8) # 1 for win and 0 for not win
        done = state.done | reward != 0 | jnp.all(current_board != 0)

        return State(
                observation=new_observation,
                current_player=(1-state.current_player),
                legal_action_mask=legal_action_mask,
                done=done
            ), reward, done
    
    @partial(jax.jit, static_argnums=0, backend='gpu')
    def get_cur_board(self, state: State) -> tuple:
        # to get current board, we only support batch size = 1 for now.
        chex.assert_shape(state.observation, (1, self._board_size, self._board_size, 17))
        current_player_piece = jnp.where(state.observation[..., 1], 2*state.current_player-1, 0)
        oppo_piece = jnp.where(state.observation[..., 0], 1-2*state.current_player, 0)
        current_board = current_player_piece + oppo_piece
        chex.assert_shape(current_board, (1, self._board_size, self._board_size))
        return current_board[0]
    
    def print_board(self, state: State):
        board = self.get_cur_board(state)
        board = jax.device_get(board) # move to CPU for printing
        row_idx = -1
        board_str = '   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4\n'
        for row in board:
            row_idx += 1
            suffix = (' ' if row_idx < 10 else '') + (str(row_idx) if row_idx < 10 else str(row_idx)) + ' '
            row_str = suffix + ' '.join([('.' if cell==0 else ('o' if cell==1 else 'x')) for cell in row]) + '\n'
            board_str += row_str
        print(board_str)



if __name__ == "__main__":
    env = Env("gomoku-15x15")

    rngkey, subkey = jax.random.split(jax.random.PRNGKey(0), 2)
    env_keys = jax.random.split(subkey, 1)
    states = env.reset(env_keys)
    print(states.observation.shape)
    print(states.current_player.shape)
    print(states.legal_action_mask.shape)
    print(states.done.shape)
    states, rewards, done = (env.step)(states, jnp.ones((1), dtype=jnp.int8))
    print(rewards.shape, done.shape)
    # print(states.observation[0,:,:,0])
    # print(rewards.shape, done.shape)
    states, rewards, done = (env.step)(states, jnp.zeros((1), dtype=jnp.int8))
    print(rewards.shape, done.shape)
    env.print_board(states)
    # print(states.observation[0,:,:,0])
    # print(rewards.shape, done.shape)
