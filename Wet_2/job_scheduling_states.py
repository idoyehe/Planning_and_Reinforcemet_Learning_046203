import numpy as np

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
    global STATE_SPACE_INDEX_DICT, STATE_SPACE_COST
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


def __build_cost_vector(state_space_size):
    global STATE_SPACE_INDEX_DICT, STATE_SPACE_COST
    cost_vector = np.zeros(shape=(state_space_size, 1))
    for state, index in STATE_SPACE_INDEX_DICT.items():
        cost_vector[index] = STATE_SPACE_COST[state]
    return cost_vector


def __build_state_space_all():
    __build_state_space([1, 2, 3, 4, 5])
    state_space_size = len(STATE_SPACE_INDEX_DICT.keys())
    STATE_SPACE_INDEX_DICT[str([])] = state_space_size
    STATE_SPACE_COST[str([])] = 0
    state_space_size += 1
    return state_space_size


STATE_SPACE_SIZE = __build_state_space_all()
COST_VECTOR = __build_cost_vector(STATE_SPACE_SIZE)
