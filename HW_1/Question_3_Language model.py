import numpy as np


def return_index(l) -> int:
    translate = {"B": 0, "K": 1, "O": 2, "-": 3}
    return translate[l]


def return_letter(index) -> str:
    translate = {0: "B", 1: "K", 2: "O", 3: "-"}
    return translate[index]


def return_probs(l_t_1, l_t) -> float:
    translate = {"B": 0, "K": 1, "O": 2, "-": 3}

    probabilities = np.array([
        [0.1, 0.325, 0.25, 0.325],
        [0.4, 0, 0.4, 0.2],
        [0.2, 0.2, 0.2, 0.4],
        [1, 0, 0, 0]])

    return probabilities[return_index(l_t), return_index(l_t_1)]


L = 5  # for length

interRes = np.ndarray((L + 2, 4), dtype=object)
interRes[1, return_index("B")] = (1, "B")
interRes[1, return_index("K")] = (0, "K")
interRes[1, return_index("O")] = (0, "O")
interRes[1, return_index("-")] = (0, "-")

for l in range(2, L + 2):
    for curr in range(0, 4):
        trans2curr = [
            return_probs(return_letter(curr), "B") * interRes[l - 1, return_index("B")][0],
            return_probs(return_letter(curr), "K") * interRes[l - 1, return_index("K")][0],
            return_probs(return_letter(curr), "O") * interRes[l - 1, return_index("O")][0],
        ]

        max_index = np.argmax(trans2curr)

        interRes[l, curr] = (max(trans2curr), interRes[l - 1, max_index][1] + return_letter(curr))

print(interRes[6, 3])
