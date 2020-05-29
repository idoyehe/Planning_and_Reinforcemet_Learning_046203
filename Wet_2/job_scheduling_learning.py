import numpy as np
from json import loads, dumps
from job_scheduling_states import \
    STATE_SPACE_INDEX_DICT, \
    STATE_SPACE_SIZE, \
    COST_VECTOR, \
    JOB_COST_MEU_DICT, \
    STATE_SPACE_COST


def simulator(current_state: str, action: int):
    current_state = loads(current_state)
    assert action in current_state
    next_state_job_done = list(current_state)
    job = action
    next_state_job_done.remove(job)
    assert dumps(next_state_job_done) in STATE_SPACE_INDEX_DICT.keys()

    job_done_prob = JOB_COST_MEU_DICT[job]["u"]
    job_not_done_prob = 1 - job_done_prob
    actual_random_next_state = \
        np.random.choice([current_state, next_state_job_done], 1, p=[job_not_done_prob, job_done_prob])[0]
    return dumps(actual_random_next_state)
