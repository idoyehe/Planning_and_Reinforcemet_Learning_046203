import numpy as np

X = "ABCBD"
Y = "ACD"
Z = "AAACD"

m = len(X)
n = len(Y)
l = len(Z)

chars = np.ndarray((m + 1, n + 1, l + 1), dtype=object)
lengths = np.zeros((m + 1, n + 1, l + 1), dtype=int)

for i in range(0, m + 1):
    for j in range(0, n + 1):
        for k in range(0, l + 1):
            chars[i, j, k] = ""

for i in range(1, m + 1):
    for j in range(1, n + 1):
        for k in range(1, l + 1):
            if X[i - 1] == Y[j - 1] == Z[k - 1]:
                lengths[i, j, k] = 1 + lengths[i - 1, j - 1, k - 1]
                chars[i, j, k] = chars[i - 1, j - 1, k - 1] + X[i - 1]
            else:
                (lengths[i, j, k], chars[i, j, k]) = max(
                    [(lengths[i - 1, j, k], chars[i - 1, j, k]), (lengths[i, j - 1, k], chars[i, j - 1, k]),
                     (lengths[i, j, k - 1], chars[i, j, k - 1])], key=lambda e: e[0])

print(lengths[m, n, l])
print(chars[m, n, l])
