def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    # remove the following line and complete the algorithm
    to_state = result[-1][0]
    from_state = prev[to_state.to_string()]
    while from_state is not None:
        for possible_action in from_state.get_actions():
            if from_state.apply_action(possible_action) == to_state:
                result.append((from_state, possible_action))
                break

        to_state = from_state
        from_state = prev[to_state.to_string()]

    result.reverse()
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan) - 1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
