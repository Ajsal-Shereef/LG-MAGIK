from collections import deque
import pickle
from enum import Enum
import numpy as np
from dqn.learning_agent.common_utils import discrete_action_to_one_hot


class TRAJECTORY_INDEX(Enum):
    STATE = 'state'
    ACTION = 'action'
    NEXT_STATE = 'next_state'
    REWARD = 'reward'
    DONE = 'done'
    INFO = 'info'
    STATE_RGB = 'state_rgb'
    NEXT_STATE_RGB = 'next_state_rgb'


def get_flatten_trajectories(trajs):
    """
    get flatten demos in which all transitions are saved in one list
    """
    flat_trajs = []
    for traj_i in range(len(trajs)):
        flat_trajs = flat_trajs + trajs[traj_i]
    return flat_trajs


def get_flatten_indexed_trajs(indexed_trajs):
    """
    get flatten demos in which all transitions are saved in one dict
    """
    # find the keys of the indexed_demos
    keys = indexed_trajs[0].keys()
    # init flat_indexed_trajs dict
    flat_indexed_trajs = {}
    for key in keys:
        flat_indexed_trajs[key] = []

    for i_traj in range(len(indexed_trajs)):
        traj = indexed_trajs[i_traj]
        for key in keys:
            flat_indexed_trajs[key] = flat_indexed_trajs[key] + traj[key]
    return flat_indexed_trajs


def get_indexed_trajs(trajectories_list):
    """
    return the indexed trajectories list
    :param trajectories_list: list of trajectories, where each trajectory is a list of tuple in
        the format of [(state, action, reward, next_state, done)]
    """
    indexed_traj_list = []
    for traj_i in range(len(trajectories_list)):
        trajectory = trajectories_list[traj_i]
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []
        info_list = []
        for t in range(len(trajectory)):
            state_list.append(trajectory[t][0])
            action_list.append(trajectory[t][1])
            reward_list.append(trajectory[t][2])
            next_state_list.append(trajectory[t][3])
            done_list.append(trajectory[t][4])
            info_list.append(trajectory[t][5])
        indexed_traj_list.append({TRAJECTORY_INDEX.STATE.value: state_list,
                                  TRAJECTORY_INDEX.ACTION.value: action_list,
                                  TRAJECTORY_INDEX.REWARD.value: reward_list,
                                  TRAJECTORY_INDEX.NEXT_STATE.value: next_state_list,
                                  TRAJECTORY_INDEX.DONE.value: done_list,
                                  TRAJECTORY_INDEX.INFO.value: info_list,
                                  })
    return indexed_traj_list


def read_expert_demo(fname, is_by_keys=False):
    """
    read human demonstration from pickle file
    :param fname: demo file (path to the file)
    :param is_by_keys: if it's set to True, then will return a list of dict [dict(trajectory), dict(trajectory)]
        which is indexed by attribute names (e.g. "state", "reward" ...)
    """
    with open(fname, 'rb') as f_traj:
        trajectories_list = pickle.load(f_traj)
        if is_by_keys:
            trajectories_list = get_indexed_trajs(trajectories_list)
    return trajectories_list


def demo_discrete_actions_to_one_hot(demo_trajectories_list, action_dim):
    """
    replace the original discrete action representation to one-hot representation (numpy.ndarray)
    all the changes are made in-place
    """
    for i_traj in range(len(demo_trajectories_list)):
        traj = demo_trajectories_list[i_traj]
        for transition_i, transition in enumerate(traj):
            action_one_hot = discrete_action_to_one_hot(transition[1], action_dim)
            traj[transition_i] = (transition[0], action_one_hot, transition[2], transition[3], transition[4], transition[5])


def get_n_step_info_from_traj(traj, n_step, gamma):
    """
    Return 1 step and n step demos.
    demo: List
    n_step: int
    gamma: float
    :return: Tuple[List, List]
    """
    assert n_step > 1

    one_step_trajs = list()
    n_step_trajs = list()
    n_step_buffer = deque(maxlen=n_step)

    for idx, transition in enumerate(traj):
        n_step_buffer.append(transition)

        if len(n_step_buffer) == n_step:
            # add a single step transition
            one_step_trajs.append(n_step_buffer[0])

            # add a multi step transition
            curr_state, action = n_step_buffer[0][:2]
            reward, next_state, done, info = get_n_step_info(n_step_buffer, gamma)

            transition = (curr_state, action, reward, next_state, done, info)
            n_step_trajs.append(transition)

    return one_step_trajs, n_step_trajs


def get_n_step_info(n_step_buffer, gamma):
    """
    Return n step reward, next state, and done.
    n_step_buffer: Deque
    gamma: float
    :return: Tuple[np.int64, np.ndarray, bool]
    """
    # info of the last transition
    reward, next_state, done, info = n_step_buffer[-1][-4:]

    reversed_transition = list(reversed(list(n_step_buffer)[:-1]))
    for i, transition in enumerate(reversed_transition):
        r, n_s, d, tran_info = transition[-4:]

        reward = r + gamma * reward * (1 - d)
        next_state, done, info = (n_s, d, tran_info) if d else (next_state, done, info)

    return reward, next_state, done, info


def extract_transitions_from_indexed_trajs(indexed_trajs):
    """
    Return a flat list of all the transitions stored in the indexed trajs
    :param indexed_trajs: list of indexed trajs
    """
    transitions = []

    obs_key = TRAJECTORY_INDEX.STATE.value
    action_key = TRAJECTORY_INDEX.ACTION.value
    next_obs_key = TRAJECTORY_INDEX.NEXT_STATE.value
    reward_key = TRAJECTORY_INDEX.REWARD.value
    done_key = TRAJECTORY_INDEX.DONE.value
    info_key = TRAJECTORY_INDEX.INFO.value
    for traj in indexed_trajs:
        for t in range(len(traj)):
            state = traj[obs_key][t]
            action = traj[action_key][t]
            next_state = traj[next_obs_key][t]
            reward = traj[reward_key][t]
            done = traj[done_key][t]
            info = traj[info_key][t]

            transitions.append([state, action, reward, next_state, done, info])
    return transitions


def extract_experiences_from_indexed_trajs(indexed_trajs):
    """
    Return a states, actions, ..., dones stored in the indexed trajs
    :param indexed_trajs: list of indexed trajectories
    """
    flatten_trajs = get_flatten_indexed_trajs(indexed_trajs)
    states = np.array(flatten_trajs[TRAJECTORY_INDEX.STATE.value])
    actions = np.array(flatten_trajs[TRAJECTORY_INDEX.ACTION.value])
    rewards = np.array(flatten_trajs[TRAJECTORY_INDEX.REWARD.value])
    next_states = np.array(flatten_trajs[TRAJECTORY_INDEX.NEXT_STATE.value])
    dones = np.array(flatten_trajs[TRAJECTORY_INDEX.DONE.value])
    infos = np.array(flatten_trajs[TRAJECTORY_INDEX.INFO.value])

    return states, actions, rewards, next_states, dones, infos


def split_fixed_length_indexed_traj(indexed_trajs, fixed_len):
    """
    Return a list indexed_traj of the same length.
    If the the length of a traj is smaller than the fixed_len, we do padding by replicating parts of the traj
    :param: indexed_trajs: list of indexed_trajs
    """
    fixed_len_indexed_trajs = []
    traj_keys = list(indexed_trajs[0].keys())
    for traj in indexed_trajs:
        traj_len = len(traj[traj_keys[0]])

        # if the length of the traj is smaller than fixed_len
        if traj_len < fixed_len:
            indexed_traj = dict()
            for key in traj_keys:
                indexed_traj[key] = []

            remain_padding_len = fixed_len
            while True:
                for key in traj_keys:
                    indexed_traj[key] = indexed_traj[key] + traj[key]
                remain_padding_len = remain_padding_len - traj_len
                if remain_padding_len - traj_len <= 0:
                    break

            for key in traj_keys:
                indexed_traj[key] = indexed_traj[key] + [traj[key][i] for i in range(0, remain_padding_len)]
            fixed_len_indexed_trajs.append(indexed_traj)
        # if the length of the traj is greater than fixed_len
        else:
            start_idx = 0
            end_idx = start_idx + fixed_len  # end idx is exclusive
            while end_idx <= traj_len:
                indexed_traj = dict()
                for key in traj_keys:
                    indexed_traj[key] = [traj[key][i] for i in range(start_idx, end_idx)]
                fixed_len_indexed_trajs.append(indexed_traj)
                start_idx += fixed_len
                end_idx = start_idx + fixed_len

            # if we need to do padding
            if start_idx < traj_len:
                padding_traj = dict()
                for key in traj_keys:
                    padding_traj[key] = [traj[key][i] for i in range(start_idx, traj_len)]

                n_repeat_transition = fixed_len - (traj_len - start_idx)
                repeat_start_idx = traj_len - n_repeat_transition
                repeat_end_idx = traj_len  # end idx is exclusive
                for key in traj_keys:
                    padding_traj[key] = padding_traj[key] + [traj[key][i] for i in
                                                             range(repeat_start_idx, repeat_end_idx)]
                fixed_len_indexed_trajs.append(padding_traj)

    return fixed_len_indexed_trajs


def split_fixed_length_traj(trajs, fixed_len):
    """
    Return a list traj of the same length.
    If the the length of a traj is smaller than the fixed_len, we do padding by replicating parts of the traj
    :param: indexed_trajs: list of indexed_trajs
    """
    fixed_len_trajs = []
    for traj in trajs:
        traj_len = len(traj)

        # if the length of the traj is smaller than fixed_len
        if traj_len < fixed_len:
            fixed_len_traj = []

            remain_padding_len = fixed_len
            while True:
                fixed_len_traj = fixed_len_traj + list(traj)
                remain_padding_len = remain_padding_len - traj_len
                if remain_padding_len - traj_len <= 0:
                    break
            fixed_len_traj = fixed_len_traj + [traj[i] for i in range(0, remain_padding_len)]
            fixed_len_trajs.append(fixed_len_traj)
        # if the length of the traj is greater than fixed_len
        else:
            start_idx = 0
            end_idx = start_idx + fixed_len  # end idx is exclusive
            while end_idx <= traj_len:
                fixed_len_trajs.append([traj[i] for i in range(start_idx, end_idx)])
                start_idx += fixed_len
                end_idx = start_idx + fixed_len

            # if we need to do padding
            if start_idx < traj_len:
                padding_traj = [traj[i] for i in range(start_idx, traj_len)]

                n_repeat_transition = fixed_len - (traj_len - start_idx)
                repeat_start_idx = traj_len - n_repeat_transition
                repeat_end_idx = traj_len  # end idx is exclusive
                padding_traj = padding_traj + [traj[i] for i in range(repeat_start_idx, repeat_end_idx)]
                fixed_len_trajs.append(padding_traj)
    return fixed_len_trajs


def stack_frames_in_trajs(trajs, frame_preprocessor, n_stack):
    stacked_trajs = []
    for traj in trajs:
        stacked_trajs.append(stack_frames_in_traj(traj, frame_preprocessor, n_stack))
    return stacked_trajs


def stack_frames_in_traj(traj, frame_preprocessor, n_stack):
    stacked_traj = []
    states_queue = None
    is_initial_state = True
    for transition in traj:
        state, action, reward, next_state, done, info = transition
        state = frame_preprocessor(state)
        state = np.squeeze(state, axis=0)
        next_state = frame_preprocessor(next_state)
        next_state = np.squeeze(next_state, axis=0)

        if is_initial_state:
            states_queue = deque(maxlen=n_stack)
            states_queue.extend([state for _ in range(n_stack)])
            is_initial_state = False

        current_stack_state = np.copy(np.stack(list(states_queue), axis=0))
        states_queue.append(next_state)
        next_stacked_state = np.copy(np.stack(list(states_queue), axis=0))
        stacked_traj.append([current_stack_state, action, reward, next_stacked_state, done, info])

        if done:
            is_initial_state = True
    return stacked_traj


def main():
    """ just for verification """
    traj = [[np.array([1]), 1, 1, np.array([2]), False],
            [np.array([2]), 2, 2, np.array([3]), True],
            [np.array([3]), 3, 3, np.array([4]), True]]
    print(stack_frames_in_traj(traj, lambda x: x, 2))


if __name__ == '__main__':
    main()



