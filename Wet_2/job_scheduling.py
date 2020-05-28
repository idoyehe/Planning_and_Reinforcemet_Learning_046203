import numpy as np
import json

JOB_COST_MEU_DICT = {
    1: {"c": 1, "u": 0.6, "cu": 0.6},
    2: {"c": 4, "u": 0.5, "cu": 2.0},
    3: {"c": 6, "u": 0.3, "cu": 1.8},
    4: {"c": 2, "u": 0.7, "cu": 1.4},
    5: {"c": 9, "u": 0.1, "cu": 0.9},
}

STATE_SPACE_INDEX_DICT = dict()
STATE_SPACE_COST = dict()


def __build_state_space(state):
    if state == []:
        return
    else:
        if str(state) not in STATE_SPACE_INDEX_DICT.keys():
            STATE_SPACE_INDEX_DICT[str(state)] = len(STATE_SPACE_INDEX_DICT.keys())
            STATE_SPACE_COST[str(state)] = sum([JOB_COST_MEU_DICT[job]["c"] for job in state])
    for j in state:
        bstate = list(state)
        bstate.remove(j)
        __build_state_space(bstate)
    return


def __build_state_space_all():
    __build_state_space([1, 2, 3, 4, 5])
    state_space_size = len(STATE_SPACE_INDEX_DICT.keys())
    STATE_SPACE_INDEX_DICT[str([])] = state_space_size
    STATE_SPACE_COST[str([])] = 0
    state_space_size += 1
    return state_space_size


def __build_cost_vector(state_space_size):
    cost_vector = np.zeros(shape=(state_space_size, 1))
    for state, index in STATE_SPACE_INDEX_DICT.items():
        cost_vector[index] = STATE_SPACE_COST[state]
    return cost_vector


STATE_SPACE_SIZE = __build_state_space_all()
COST_VECTOR = __build_cost_vector(STATE_SPACE_SIZE)


def __build_transition_matrix_by_policy(policy):
    prob_trans = np.zeros(shape=(STATE_SPACE_SIZE, STATE_SPACE_SIZE))
    for from_state, index in STATE_SPACE_INDEX_DICT.items():
        from_state = json.loads(from_state)
        if len(from_state) > 0:
            chosen_job = int(policy[index])
            to_state = list(from_state)
            assert chosen_job in from_state
            to_state.remove(chosen_job)
            job_meu = JOB_COST_MEU_DICT[chosen_job]["u"]
            i = STATE_SPACE_INDEX_DICT[str(from_state)]
            j = STATE_SPACE_INDEX_DICT[str(to_state)]
            prob_trans[i][j] = job_meu
            prob_trans[i][i] = 1 - job_meu
    return prob_trans


def calculate_value_function_by_policy(policy):
    assert policy.shape == COST_VECTOR.shape
    transition_matrix = __build_transition_matrix_by_policy(policy)
    value_function = np.linalg.inv(np.eye(N=STATE_SPACE_SIZE) - transition_matrix) @ COST_VECTOR
    value_function[STATE_SPACE_INDEX_DICT[str([])]] = 0
    return value_function


def b_task_greedy_policy_by_cost():
    policy = np.zeros(shape=(STATE_SPACE_SIZE, 1))

    for state, index in STATE_SPACE_INDEX_DICT.items():
        state = json.loads(state)
        if len(state) > 0:
            jobs_and_cost = [(job, JOB_COST_MEU_DICT[job]["c"]) for job in state]
            policy[index] = max(jobs_and_cost, key=lambda x: x[1])[0]
        else:
            policy[index] = -1

    return policy, calculate_value_function_by_policy(policy)


def policy_iteration():
    halt_flag = False
    current_policy = b_task_greedy_policy_by_cost()[0]
    while not halt_flag:
        current_policy_value_function = calculate_value_function_by_policy(current_policy)
        # update policy respect to each state
        next_policy = np.zeros(shape=current_policy.shape)
        for state, index in STATE_SPACE_INDEX_DICT.items():
            cost_of_state = STATE_SPACE_COST[state]
            state = json.loads(state)
            current_min_value = float("inf")
            current_best_action = -1
            for job in state:
                state_tag = list(state)
                state_tag.remove(job)
                value = cost_of_state + float(current_policy_value_function[STATE_SPACE_INDEX_DICT[str(state_tag)]])
                if value < current_min_value:
                    current_min_value = value
                    current_best_action = job
            next_policy[index] = current_best_action
        halt_flag = np.array_equal(current_policy, next_policy)

    return current_policy


optimal = policy_iteration()
pass
