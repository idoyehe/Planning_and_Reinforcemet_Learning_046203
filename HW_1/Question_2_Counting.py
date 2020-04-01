import numpy as np

N = 12
X = 800

interRes = np.zeros((N + 1, X + 1))

for x in range(1, X + 1):
    interRes[1, x] = 1

for n in range(2, N + 1):
    for x in range(1, X + 1):
        interRes[n, x] = interRes[n, x - 1] + interRes[n - 1, x - 1]

print(int(interRes[12, 800]))
