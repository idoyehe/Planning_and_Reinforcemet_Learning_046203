import numpy as np
from cartpole_cont import CartPoleContEnv
from json import dump


def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    moment_of_inertia = pole_mass * (pole_length ** 2)

    A_bar = np.matrix([[0, 1, 0, 0],
                       [0, 0, ((pole_mass * g) / cart_mass), 0],
                       [0, 0, 0, 1],
                       [0, 0, (g / pole_length) * (1 + (pole_mass / cart_mass)), 0]])
    A = np.identity(4) + (dt * A_bar)

    return A


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    B_bar = np.matrix([[0],
                       [1 / cart_mass],
                       [0],
                       [1 / (cart_mass * pole_length)]])

    B = B_bar * dt

    return B


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    A_T = A.T
    B_T = B.T

    Q = np.matrix([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = np.matrix([1])

    Ps = [Q]
    horizon = cart_pole_env.planning_steps
    Ks = []
    for _ in range(horizon):
        p_t_1 = Ps[-1]
        common_expression = ((R + B_T * p_t_1 * B).I) * B_T * p_t_1 * A
        prev_k = -common_expression
        prev_p = Q + (A_T * p_t_1 * A) - (A_T * p_t_1 * B * common_expression)
        Ks.append(prev_k)
        Ps.append(prev_p)

    Ps.reverse()
    Ks.reverse()

    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]

    for controller_gain in Ks:
        current_state = xs[-1]
        current_action = controller_gain * current_state
        next_state = (A * current_state) + (B * current_action)

        us.append(current_action)
        xs.append(next_state)

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


if __name__ == '__main__':
    env = CartPoleContEnv(initial_theta=np.pi * 0.1)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.4)

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    theta_records = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        theta_records.append(actual_theta)
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        # actual_action = predicted_action # feedforward control
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))

    # with open('unstable_theta_records_feedforward.json', 'w') as f:
    #     dump(theta_records, f)
    #     f.close()
