import numpy as np
from mountain_car_with_data_collection import MountainCarWithResetEnv
from q_learn_mountain_car import Solver, run_episode
from collections import defaultdict
import matplotlib.pyplot as plt


def run_q_learning_training(seed, epsilon=0.1, max_episodes=1000):
    env = MountainCarWithResetEnv()
    np.random.seed(seed)
    env.seed(seed)

    gamma = 0.999
    learning_rate = 0.01

    max_episodes = max_episodes
    solver = Solver(
        # learning parameters
        gamma=gamma, learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
    )
    train_statistics = defaultdict(list)

    bellman_error = list()
    bellman_error_index = 100
    for episode_index in range(1, max_episodes + 1):
        episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon)
        bellman_error.append(mean_delta)

        print(f'After {episode_index}, reward = {episode_gain}, epsilon {epsilon}, average error {mean_delta}')
        env.reset()
        init_state = env.state
        phi_st_0 = solver.get_state_action_features(init_state, 0)
        phi_st_1 = solver.get_state_action_features(init_state, 1)
        phi_st_2 = solver.get_state_action_features(init_state, 2)
        Q_st_0 = phi_st_0.transpose() @ solver.theta
        Q_st_1 = phi_st_1.transpose() @ solver.theta
        Q_st_2 = phi_st_2.transpose() @ solver.theta

        train_statistics["init_state"].append(max(Q_st_0, Q_st_1, Q_st_2))
        train_statistics["reward"].append(episode_gain)

        # if episode_index % 100 == 99:
        #     train_statistics["bellman_error"].append(np.mean(bellman_error))
        #     train_statistics["bellman_error_index"].append(bellman_error_index)
        #     bellman_error_index += 100
        #     bellman_error = list()

        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
            mean_test_gain = np.mean(test_gains)
            train_statistics["performance"].append(mean_test_gain)

            print(f'tested 10 episodes: mean gain is {mean_test_gain}')
            if mean_test_gain >= -75.:
                print(f'solved in {episode_index} episodes')
                break

    return train_statistics


def plot_success_rate_vs_lspi_iteration(y_data, y_label, x_data=None):
    if x_data is not None:
        plt.plot(x_data, y_data)
    else:
        plt.plot(y_data)
    plt.ylabel(y_label)
    plt.xlabel('# Episodes')
    plt.title('{} Vs. Training Episodes'.format(y_label))
    plt.show()


def reward_per_training_episode_vs_training_episodes(l1):
    plot_success_rate_vs_lspi_iteration(l1, "Total Reward")


def performance_per_training_episode_vs_training_episodes(l1):
    plot_success_rate_vs_lspi_iteration(l1, "Performance")


def v_s0_per_training_episode_vs_training_episodes(l1):
    plot_success_rate_vs_lspi_iteration(l1, "Init State Value")


def error_per_training_episode_vs_training_episodes(l1, x_data):
    plot_success_rate_vs_lspi_iteration(l1, "Bellman Error", x_data)


def plot_statistics_by_seed(seed):
    train_statistics = run_q_learning_training(seed)
    reward_per_training_episode_vs_training_episodes(train_statistics["reward"])
    performance_per_training_episode_vs_training_episodes(train_statistics["performance"])
    v_s0_per_training_episode_vs_training_episodes(train_statistics["init_state"])
    error_per_training_episode_vs_training_episodes(train_statistics["bellman_error"], train_statistics["bellman_error_index"])


def plot_statistics_by_epsilon(seed, epsilon):
    train_statistics = run_q_learning_training(seed, epsilon)
    reward_per_training_episode_vs_training_episodes(train_statistics["reward"])


if __name__ == "__main__":
    # plot_statistics_by_seed(100)
    # plot_statistics_by_seed(200)
    # plot_statistics_by_seed(300)
    plot_statistics_by_epsilon(123, 0.01)
    plot_statistics_by_epsilon(123, 0.3)
    plot_statistics_by_epsilon(123, 0.5)
    plot_statistics_by_epsilon(123, 0.75)
    plot_statistics_by_epsilon(123, 1)
