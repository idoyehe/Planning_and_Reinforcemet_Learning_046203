import numpy as np
import matplotlib.pyplot as plt
from json import loads, dumps
from collections import defaultdict
from job_scheduling_states import \
    STATE_SPACE_INDEX_DICT, \
    STATE_SPACE_SIZE, \
    COST_VECTOR, \
    JOB_COST_MEU_DICT, \
    STATE_SPACE_COST, \
    INIT_STATE
from job_scheduling import b_task_greedy_policy_by_cost
from statistics import mean


def simulator(current_state: str, action: int):
    current_state_cost = STATE_SPACE_COST[current_state]
    current_state = loads(current_state)
    if len(current_state) > 0:
        assert action in current_state
        next_state_job_done = list(current_state)
        job = action
        next_state_job_done.remove(job)
        assert dumps(next_state_job_done) in STATE_SPACE_INDEX_DICT.keys()

        job_done_prob = JOB_COST_MEU_DICT[job]["u"]
        job_not_done_prob = 1 - job_done_prob
        actual_random_next_state = \
            np.random.choice([current_state, next_state_job_done], 1, p=[job_not_done_prob, job_done_prob])[0]

        actual_random_next_state = dumps(actual_random_next_state)
        return current_state_cost, actual_random_next_state, False
    else:
        return current_state_cost, dumps(current_state), True


def TD_0(episodes, alpha_n_method):
    cost_greedy_policy, cost_greedy_values = b_task_greedy_policy_by_cost()
    value_function = np.zeros(shape=(len(STATE_SPACE_INDEX_DICT.keys()), 1))

    max_norm_delta = list()
    s0_delta = list()
    visited_in_state_counter = defaultdict(int)

    for _ in range(episodes):
        current_state = np.random.choice(list(STATE_SPACE_INDEX_DICT.keys()))
        current_state_index = STATE_SPACE_INDEX_DICT[current_state]
        visited_in_state_counter[current_state] += 1
        done = False
        while not done:
            cost_state, next_state, done = simulator(current_state, int(cost_greedy_policy[current_state_index]))
            next_state_index = STATE_SPACE_INDEX_DICT[next_state]
            visited_in_state_counter[next_state] += 1

            if alpha_n_method == 1:
                alpha_n = 1 / visited_in_state_counter[current_state]
            elif alpha_n_method == 2:
                alpha_n = 0.01
            else:
                alpha_n = 10 / (100 + visited_in_state_counter[current_state])

            d_n = cost_state + float(value_function[next_state_index]) - float(value_function[current_state_index])
            value_function[current_state_index] = value_function[current_state_index] + alpha_n * d_n

            current_state = next_state
            current_state_index = next_state_index

            max_norm_delta.append(np.max(np.abs(value_function - cost_greedy_values)))
            s0_delta.append(np.abs(value_function[0] - cost_greedy_values[0])[0])
    return max_norm_delta, s0_delta


def TD_lamda(n_iter, lamda):
    cost_greedy_policy, cost_greedy_values = b_task_greedy_policy_by_cost()
    max_norm_delta_exp = list()
    s0_delta_exp = list()

    for _ in range(20):
        visited_in_state_counter = defaultdict(int)
        current_state = INIT_STATE
        visited_in_state_counter[current_state] += 1
        value_function = np.zeros(shape=(len(STATE_SPACE_INDEX_DICT.keys()), 1))
        eligibility = np.zeros(value_function.shape)
        max_norm_delta = list()
        s0_delta = list()
        current_state = INIT_STATE
        current_state_index = STATE_SPACE_INDEX_DICT[current_state]
        visited_in_state_counter[current_state] += 1
        for _ in range(n_iter):
            cost_state, next_state, done = simulator(current_state, int(cost_greedy_policy[current_state_index]))
            next_state_index = STATE_SPACE_INDEX_DICT[next_state]
            visited_in_state_counter[next_state] += 1

            eligibility *= lamda * 1
            eligibility[current_state_index] += 1.0

            alpha_n = 10 / (100 + visited_in_state_counter[current_state])

            td_error = cost_state + float(value_function[next_state_index]) - float(
                value_function[current_state_index])
            value_function = value_function + alpha_n * td_error * eligibility

            if done:
                current_state = INIT_STATE
                current_state_index = STATE_SPACE_INDEX_DICT[current_state]
            else:
                current_state = next_state
                current_state_index = next_state_index

            max_norm_delta.append(np.max(np.abs(value_function - cost_greedy_values)))
            s0_delta.append(np.abs(value_function[0] - cost_greedy_values[0])[0])

        max_norm_delta_exp.append(max_norm_delta)
        s0_delta_exp.append(s0_delta)
    max_norm_delta_exp = list(map(mean, zip(*max_norm_delta_exp)))
    s0_delta_exp = list(map(mean, zip(*s0_delta_exp)))
    return max_norm_delta_exp, s0_delta_exp


def __plot_deltas(max_norm_delta, s0_delta):
    fig, ax = plt.subplots(1, 1)
    ax.plot(list(range(len(max_norm_delta))), max_norm_delta, label="Max Norm Delta")
    ax.plot(list(range(len(s0_delta))), s0_delta, label="Initial State Delta")

    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Delta")
    ax.set_title("Delta Vs. Iterations")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    episodes = 10000
    # alpha n is 1/ number of visits
    max_norm_delta, s0_delta = TD_0(episodes, 1)
    __plot_deltas(max_norm_delta, s0_delta)

    # alpha n is 0.01 constant
    max_norm_delta, s0_delta = TD_0(episodes, 2)
    __plot_deltas(max_norm_delta, s0_delta)

    # alpha n is 10 / (100 + number of visits)
    max_norm_delta, s0_delta = TD_0(episodes, 3)
    __plot_deltas(max_norm_delta, s0_delta)

    # alpha n is 10 / (100 + number of visits) of visits and lambda is 0
    max_norm_delta, s0_delta = TD_lamda(episodes, 0)
    __plot_deltas(max_norm_delta, s0_delta)

    # alpha n is 10 / (100 + number of visits) of visits and lambda is 0
    max_norm_delta, s0_delta = TD_lamda(episodes, 0.5)
    __plot_deltas(max_norm_delta, s0_delta)

    # alpha n is 10 / (100 + number of visits) of visits and lambda is 0
    max_norm_delta, s0_delta = TD_lamda(episodes, 0.9)
    __plot_deltas(max_norm_delta, s0_delta)
