import numpy as np

N = 12
X = 800

intermediate_results = np.zeros((N + 1, X + 1))

for x in range(1, X + 1):
    intermediate_results[1, x] = 1

for x in range(1, X + 1):
    for n in range(1, N + 1):
        for i in range(1, x):
            intermediate_results[n, x] += intermediate_results[n - 1, x - i]

print(int(intermediate_results[12, 800]))
