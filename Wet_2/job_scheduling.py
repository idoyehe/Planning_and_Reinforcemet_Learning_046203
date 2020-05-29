import numpy as np
import matplotlib.pyplot as plt
from json import loads, dumps
from job_scheduling_states import \
    STATE_SPACE_INDEX_DICT, \
    STATE_SPACE_SIZE, \
    COST_VECTOR, \
    JOB_COST_MEU_DICT, \
    STATE_SPACE_COST


def __build_transition_matrix_by_policy(policy):
    prob_trans = np.zeros(shape=(STATE_SPACE_SIZE, STATE_SPACE_SIZE))
    for from_state, index in STATE_SPACE_INDEX_DICT.items():
        from_state = loads(from_state)
        if len(from_state) > 0:
            chosen_job = int(policy[index])
            to_state = list(from_state)
            assert chosen_job in from_state
            to_state.remove(chosen_job)
            job_meu = JOB_COST_MEU_DICT[chosen_job]["u"]
            i = STATE_SPACE_INDEX_DICT[dumps(from_state)]
            j = STATE_SPACE_INDEX_DICT[dumps(to_state)]
            prob_trans[i][j] = job_meu
            prob_trans[i][i] = 1 - job_meu
    return prob_trans


def calculate_value_function_by_policy(policy):
    assert policy.shape == COST_VECTOR.shape
    transition_matrix = __build_transition_matrix_by_policy(policy)
    value_function = np.linalg.inv(np.eye(N=STATE_SPACE_SIZE) - transition_matrix) @ COST_VECTOR
    value_function[STATE_SPACE_INDEX_DICT[dumps([])]] = 0
    return value_function


def b_task_greedy_policy_by_cost(plot=False):
    policy = np.zeros(shape=(STATE_SPACE_SIZE, 1))

    # generating greedy policy
    for state, index in STATE_SPACE_INDEX_DICT.items():
        state = loads(state)
        if len(state) > 0:
            jobs_and_cost = [(job, JOB_COST_MEU_DICT[job]["c"]) for job in state]
            policy[index] = max(jobs_and_cost, key=lambda x: x[1])[0]
        else:
            policy[index] = -1
    policy_values = calculate_value_function_by_policy(policy)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(list(STATE_SPACE_INDEX_DICT.values()), policy_values)
        # Set number of ticks for x-axis
        ax.set_xticks(list(STATE_SPACE_INDEX_DICT.values()))
        # Set ticks labels for x-axis
        ax.set_xticklabels(list(STATE_SPACE_INDEX_DICT.keys()), rotation='vertical', fontsize=12)
        ax.set_xlabel("State")
        ax.set_ylabel("Value")
        ax.set_title("Value Function of PI_C Vs. State")
        plt.show()
    return policy, policy_values


def __optimal_policy():
    opt_policy = np.zeros(shape=(STATE_SPACE_SIZE, 1))
    # generating optimal policy for validation
    for state, index in STATE_SPACE_INDEX_DICT.items():
        state = loads(state)
        if len(state) > 0:
            jobs_and_cost = [(job, JOB_COST_MEU_DICT[job]["cu"]) for job in state]
            opt_policy[index] = max(jobs_and_cost, key=lambda x: x[1])[0]
        else:
            opt_policy[index] = -1
    return opt_policy


def policy_iteration():
    halt_flag = False
    current_policy = b_task_greedy_policy_by_cost()[0]
    s_0_values = list()

    while not halt_flag:
        current_policy_value_function = calculate_value_function_by_policy(current_policy)
        s_0_values.append(float(current_policy_value_function[0]))
        # update policy respect to each state
        next_policy = np.zeros(shape=current_policy.shape)
        for state, state_index in STATE_SPACE_INDEX_DICT.items():
            cost_of_state = STATE_SPACE_COST[state]
            state = loads(state)
            current_min_value = float("inf")
            current_best_action = -1
            for job in state:
                state_tag = list(state)  # copy
                state_tag.remove(job)
                assert dumps(state_tag) in STATE_SPACE_INDEX_DICT.keys()
                state_tag_index = STATE_SPACE_INDEX_DICT[dumps(state_tag)]

                job_done_prob = JOB_COST_MEU_DICT[job]["u"]
                job_not_done_prob = 1 - job_done_prob

                new_value = \
                    cost_of_state + \
                    job_done_prob * float(current_policy_value_function[state_tag_index]) + \
                    job_not_done_prob * float(current_policy_value_function[state_index])

                if new_value < current_min_value:
                    current_min_value = new_value
                    current_best_action = job
            next_policy[state_index] = current_best_action
        halt_flag = np.array_equal(current_policy, next_policy)
        current_policy = next_policy

    assert np.array_equal(current_policy, __optimal_policy())
    fig, ax = plt.subplots(1, 1)
    ax.plot(list(range(len(s_0_values))), s_0_values)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("State 0 value Vs. policy iteration")
    plt.show()
    return current_policy


def e_task_compare_policies(optimal_policy):
    assert np.array_equal(optimal_policy, __optimal_policy())
    opt_value_function = calculate_value_function_by_policy(optimal_policy)
    cost_greedy_policy, cost_greedy_values = b_task_greedy_policy_by_cost()
    fig, ax = plt.subplots(1, 1)
    ax.plot(list(STATE_SPACE_INDEX_DICT.values()), opt_value_function, label="Optimal Value Function")
    ax.plot(list(STATE_SPACE_INDEX_DICT.values()), cost_greedy_values, label="Cost Greedy Value Function")
    # Set number of ticks for x-axis
    ax.set_xticks(list(STATE_SPACE_INDEX_DICT.values()))
    # Set ticks labels for x-axis
    ax.set_xticklabels(list(STATE_SPACE_INDEX_DICT.keys()), rotation='vertical', fontsize=12)
    ax.set_xlabel("State")
    ax.set_ylabel("Value Function")
    ax.set_title("Value Function Vs. States")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    b_task_greedy_policy_by_cost(True)
    optimal = policy_iteration()
    e_task_compare_policies(optimal)
