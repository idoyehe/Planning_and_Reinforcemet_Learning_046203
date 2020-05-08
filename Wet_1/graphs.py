from json import load
from sys import argv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    theta_records = None
    json_filename = argv[1]
    with open(json_filename, 'r') as f:
        theta_records = load(f)
        f.close()
    assert isinstance(theta_records, list)
    # Plot the data
    plt.plot(range(1, 601), theta_records)

    # Add a titles
    plt.xlabel("# Iteration")
    plt.ylabel("Theta Value")
    plt.title("Theta Values Vs. Iterations")

    # save the plot
    plt.savefig("{}.png".format(json_filename.split(".")[0]))
