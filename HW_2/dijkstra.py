from puzzle import *
from planning_utils import *
import heapq
import datetime


def dijkstra(puzzle):
    '''
    apply dijkstra to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # the fringe is the queue to pop items from
    fringe = [(0, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    while len(fringe) > 0:
        cost, u_state = heapq.heappop(fringe)
        u_state_string = u_state.to_string()
        if u_state_string in concluded:
            continue

        concluded.add(u_state.to_string())
        if u_state == goal_state:
            break

        for possible_action in u_state.get_actions():
            alt = distances[u_state_string] + 1
            v_state = u_state.apply_action(possible_action)
            v_state_string = v_state.to_string()
            if v_state_string not in distances or alt < distances[v_state_string]:
                distances[v_state_string] = alt
                prev[v_state_string] = u_state
                heapq.heappush(fringe, (alt, v_state))
    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = dijkstra(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now() - solution_start_time))
