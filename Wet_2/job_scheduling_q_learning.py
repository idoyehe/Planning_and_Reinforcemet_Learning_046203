import numpy as np
import matplotlib.pyplot as plt
from json import loads, dumps
from collections import defaultdict
from job_scheduling_states import *
from job_scheduling import calculate_value_function_by_policy, calculate_value_function_of_optimal_policy
from statistics import mean


def createEpsilonGreedyPolicy(Q_S_A, epsilon):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """

    def policyFunction(state):
        num_actions = len(loads(state))
        action_probabilities = dict()
        best_action = min(Q_S_A[state], key=Q_S_A[state].get)
        if num_actions > 1:
            action_probabilities[best_action] = (1.0 - epsilon)
            for action in Q_S_A[state].keys():
                if action != best_action:
                    action_probabilities[action] = epsilon / (num_actions - 1)
        else:
            action_probabilities[best_action] = 1

        return action_probabilities

    return policyFunction


def createGreedyPolicy(Q_S_A):
    greedy_policy = np.zeros(shape=(STATE_SPACE_SIZE, 1))
    greedy_policy[-1] = -1
    for state, state_index in STATE_SPACE_INDEX_DICT.items():
        if len(loads(state)) > 0:
            best_action = min(Q_S_A[state], key=Q_S_A[state].get)
            greedy_policy[state_index] = best_action
    return greedy_policy


def q_learning(epsilon, num_iteration, alpha_n_method):
    optimal_value_function = calculate_value_function_of_optimal_policy()

    Q_S_A = defaultdict(dict)
    for state in STATE_SPACE_INDEX_DICT.keys():
        for action in loads(state):
            Q_S_A[state][action] = 0
        if len(loads(state)) == 0:
            Q_S_A[state][-1] = 0
    epsilon_greedy_policy = createEpsilonGreedyPolicy(Q_S_A, epsilon)

    visited_in_state_counter = defaultdict(int)
    current_state = INIT_STATE
    visited_in_state_counter[current_state] += 1

    max_norm_delta = list()
    s0_delta = list()

    # For every episode
    for ith_episode in range(num_iteration):
        # get probabilities of all actions from current state
        action_probabilities = epsilon_greedy_policy(current_state)

        # choose action according to the probability distribution
        action = np.random.choice(list(action_probabilities.keys()), p=list(action_probabilities.values()))

        # take action and get reward, transit to next state
        current_cost_state, next_state, done = simulator(current_state, action)
        visited_in_state_counter[next_state] += 1

        next_state_index = STATE_SPACE_INDEX_DICT[next_state]

        best_next_action = int(createGreedyPolicy(Q_S_A)[next_state_index])
        td_target = current_cost_state + Q_S_A[next_state][best_next_action]
        td_delta = td_target - Q_S_A[current_state][action]

        if alpha_n_method == 1:
            alpha_n = 1 / visited_in_state_counter[current_state]
        elif alpha_n_method == 2:
            alpha_n = 0.01
        else:
            alpha_n = 10 / (100 + visited_in_state_counter[current_state])

        Q_S_A[current_state][action] += alpha_n * td_delta
        current_greedy_policy_value = calculate_value_function_by_policy(createGreedyPolicy(Q_S_A))

        max_norm_delta.append(np.max(np.abs(optimal_value_function - current_greedy_policy_value)))
        s0_delta.append(np.abs(optimal_value_function[0] - Q_S_A[INIT_STATE][int(createGreedyPolicy(Q_S_A)[0])])[0])

        # done is True if episode terminated
        if done:
            current_state = INIT_STATE
            visited_in_state_counter[current_state] += 1
        else:
            current_state = next_state
    return max_norm_delta, s0_delta


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
    num_iteration = 100000
    # alpha n is 1/ number of visits
    max_norm_delta, s0_delta = q_learning(epsilon=0.1, num_iteration=num_iteration, alpha_n_method=1)
    __plot_deltas(max_norm_delta, s0_delta)

    # alpha n is 0.01
    max_norm_delta, s0_delta = q_learning(epsilon=0.1, num_iteration=num_iteration, alpha_n_method=2)
    __plot_deltas(max_norm_delta, s0_delta)

    # # alpha n is 10 / (100 + number of visits)
    # max_norm_delta, s0_delta = q_learning(epsilon=0.1, num_iteration=num_iteration, alpha_n_method=3)
    # __plot_deltas(max_norm_delta, s0_delta)
    #
    # # alpha n is 10 / (100 + number of visits) with epsilon 0.01
    # max_norm_delta, s0_delta = q_learning(epsilon=0.01, num_iteration=num_iteration, alpha_n_method=3)
    # __plot_deltas(max_norm_delta, s0_delta)
