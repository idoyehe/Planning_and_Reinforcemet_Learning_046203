from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor
import numpy as np
import matplotlib.pyplot as plt
from data_collector import DataCollector


# part 1 task 2
def plotting_the_state_features():
    rbf = RadialBasisFunctionExtractor(number_of_kernels_per_dim=[10, 8])
    position_v, velocity_v, n = __discretize_state_space(200)
    states = np.zeros(shape=(n * n, 2))

    state_index_map = list()

    index = 0
    for i in range(n):
        for j in range(n):
            states[index, 0] = position_v[i, j]
            states[index, 1] = velocity_v[i, j]
            index += 1
            state_index_map.append((i, j))
    features = rbf.encode_states_with_radial_basis_functions(states)

    first_feature = features[:, 0]
    second_feature = features[:, 1]

    first_feature_matrix = np.zeros(shape=(n, n))
    for index, value in enumerate(first_feature):
        i, j = state_index_map[index]
        first_feature_matrix[i, j] = value

    second_feature_matrix = np.zeros(shape=(n, n))
    for index, value in enumerate(second_feature):
        i, j = state_index_map[index]
        second_feature_matrix[i, j] = value

    ax = plt.axes(projection='3d')
    ax.plot_surface(position_v, velocity_v, first_feature_matrix, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('First Feature Vs. States')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('First Feature Value')
    plt.show()

    ax = plt.axes(projection='3d')
    ax.plot_surface(position_v, velocity_v, second_feature_matrix, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Second Feature Vs. States')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Second Feature Value')
    plt.show()


def __discretize_state_space(n: int = 100):
    states = MountainCarWithResetEnv().observation_space
    positions = np.linspace(states.low[0], states.high[0], n)
    velocities = np.linspace(states.low[1], states.high[1], n)

    position_v, velocity_v = np.meshgrid(positions, velocities, sparse=False, indexing='ij')

    return position_v, velocity_v, n


# part 2 task 2
def mean_std_of_states():
    env = MountainCarWithResetEnv()
    samples_to_collect = 100000
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    all_states = np.concatenate((states, next_states))
    states_mean = np.mean(all_states, axis=0)
    states_std = np.std(all_states, axis=0)
    print("states_mean: {}, states_std: {}".format(states_mean, states_std))


if __name__ == "__main__":
    plotting_the_state_features()
    mean_std_of_states()
