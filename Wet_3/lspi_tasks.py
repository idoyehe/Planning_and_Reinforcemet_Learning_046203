import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
from lspi import compute_lspi_iteration
import matplotlib.pyplot as plt
from operator import add


def training_the_model(samples_to_collect=100000, seed=100):
    number_of_kernels_per_dim = [10, 8]
    gamma = 0.999
    w_updates = 20
    evaluation_number_of_games = 50
    evaluation_max_steps_per_game = 300
    np.random.seed(seed)

    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print(f'Data Success Rate {data_success_rate}')
    # standardize data
    data_transformer = DataTransformer()
    data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    states = data_transformer.transform_states(states)
    next_states = data_transformer.transform_states(next_states)
    # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # encode all states:
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    # set a new linear policy
    linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
    # but set the weights as random
    linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
    # start an object that evaluates the success rate over time
    evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)

    success_rate_vs_iteration = list()

    for lspi_iteration in range(w_updates):
        print(f'Starting LSPI iteration {lspi_iteration}')

        new_w = compute_lspi_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        )
        norm_diff = linear_policy.set_w(new_w)

        success_rate = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)

        success_rate_vs_iteration.append(success_rate)

        if norm_diff < 0.00001:
            break

    print('LSPI Done')
    return success_rate_vs_iteration


def plot_success_rate_vs_lspi_iteration(success_rate_vs_iteration, samples=100000):
    iterations = range(1, len(success_rate_vs_iteration) + 1)
    plt.plot(success_rate_vs_iteration)
    plt.ylabel('Success Rate')
    plt.xlabel('# Iterations')
    plt.title('Success Rate Vs. Iteration; Samples: {}'.format(samples))
    plt.xticks(iterations)
    plt.show()


def __averged_3_lists(l1, l2, l3):
    avg = list(map(add, l1, l2))
    avg = list(map(add, avg, l3))
    avg = list(map(lambda x: x / 3, avg))
    return avg


def plotting_success_rate_vs_samples():
    samples_to_collect = 25000
    success_rate_vs_iteration_25000_0 = training_the_model(samples_to_collect, 123)
    success_rate_vs_iteration_25000_1 = training_the_model(samples_to_collect, 234)
    success_rate_vs_iteration_25000_2 = training_the_model(samples_to_collect, 345)
    avg_25000 = __averged_3_lists(success_rate_vs_iteration_25000_0, success_rate_vs_iteration_25000_1,
                                  success_rate_vs_iteration_25000_2)
    plot_success_rate_vs_lspi_iteration(avg_25000, samples_to_collect)

    samples_to_collect *= 2
    success_rate_vs_iteration_50000_0 = training_the_model(samples_to_collect, 123)
    success_rate_vs_iteration_50000_1 = training_the_model(samples_to_collect, 234)
    success_rate_vs_iteration_50000_2 = training_the_model(samples_to_collect, 345)
    avg_50000 = __averged_3_lists(success_rate_vs_iteration_50000_0, success_rate_vs_iteration_50000_1,
                                  success_rate_vs_iteration_50000_2)
    plot_success_rate_vs_lspi_iteration(avg_50000, samples_to_collect)

    samples_to_collect *= 2
    success_rate_vs_iteration_100000_0 = training_the_model(samples_to_collect, 123)
    success_rate_vs_iteration_100000_1 = training_the_model(samples_to_collect, 234)
    success_rate_vs_iteration_100000_2 = training_the_model(samples_to_collect, 345)
    avg_100000 = __averged_3_lists(success_rate_vs_iteration_100000_0, success_rate_vs_iteration_100000_1,
                                   success_rate_vs_iteration_100000_2)
    plot_success_rate_vs_lspi_iteration(avg_100000, samples_to_collect)

    samples_to_collect *= 2
    success_rate_vs_iteration_200000_0 = training_the_model(samples_to_collect, 123)
    success_rate_vs_iteration_200000_1 = training_the_model(samples_to_collect, 234)
    success_rate_vs_iteration_200000_2 = training_the_model(samples_to_collect, 345)
    avg_200000 = __averged_3_lists(success_rate_vs_iteration_200000_0, success_rate_vs_iteration_200000_1,
                                   success_rate_vs_iteration_200000_2)
    plot_success_rate_vs_lspi_iteration(avg_200000, samples_to_collect)


if __name__ == "__main__":
    plotting_success_rate_vs_samples()
