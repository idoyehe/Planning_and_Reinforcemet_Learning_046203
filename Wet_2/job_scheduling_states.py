import numpy as np
from json import dumps, loads

JOB_COST_MEU_DICT = {
    1: {"c": 1, "u": 0.6, "cu": 0.6},
    2: {"c": 4, "u": 0.5, "cu": 2.0},
    3: {"c": 6, "u": 0.3, "cu": 1.8},
    4: {"c": 2, "u": 0.7, "cu": 1.4},
    5: {"c": 9, "u": 0.1, "cu": 0.9},
}

STATE_SPACE_INDEX_DICT = dict()
STATE_SPACE_COST = dict()
INIT_STATE = dumps(list(range(1, 6)))


def __build_state_space(state):
    global STATE_SPACE_INDEX_DICT, STATE_SPACE_COST
    if state == []:
        return
    else:
        if dumps(state) not in STATE_SPACE_INDEX_DICT.keys():
            STATE_SPACE_INDEX_DICT[dumps(state)] = len(STATE_SPACE_INDEX_DICT.keys())
            STATE_SPACE_COST[dumps(state)] = sum([JOB_COST_MEU_DICT[job]["c"] for job in state])
    for j in state:
        bstate = list(state)
        bstate.remove(j)
        __build_state_space(bstate)
    return


def __build_cost_vector(state_space_size):
    global STATE_SPACE_INDEX_DICT, STATE_SPACE_COST
    cost_vector = np.zeros(shape=(state_space_size, 1))
    for state, index in STATE_SPACE_INDEX_DICT.items():
        cost_vector[index] = STATE_SPACE_COST[state]
    return cost_vector


def __build_state_space_all():
    jobs = list(JOB_COST_MEU_DICT.keys())
    __build_state_space(jobs)
    state_space_size = len(STATE_SPACE_INDEX_DICT.keys())
    STATE_SPACE_INDEX_DICT[dumps([])] = state_space_size
    STATE_SPACE_COST[dumps([])] = 0
    state_space_size += 1
    return state_space_size, len(jobs)


STATE_SPACE_SIZE, ACTION_SPACE_SIZE = __build_state_space_all()
COST_VECTOR = __build_cost_vector(STATE_SPACE_SIZE)


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
