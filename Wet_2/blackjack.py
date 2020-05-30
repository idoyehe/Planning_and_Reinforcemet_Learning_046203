import numpy as np
import matplotlib.pyplot as plt
from dealer_probabilty import calc_first_card_end_sum_probability, PROBABILITY_DICT, DEALER_BUSTED

Nx = 22
Ny = 12

p = calc_first_card_end_sum_probability()
r = np.zeros(shape=(Nx, Ny))

for y in range(2, Ny):
    for x in range(4, Nx):
        r[x, y] = p[y][DEALER_BUSTED] + sum(p[y, 0:x]) - sum(p[y, x + 1:DEALER_BUSTED])

v_f = np.zeros(shape=(Nx + 1, Ny))
actions = np.zeros(shape=(Nx + 1, Ny))
v_f[Nx][:] = 1
v_prev = v_f
halt_flag = False

while not halt_flag:
    for x in range(4, Nx):
        for y in range(Ny):
            sum_for_hits = 0
            for hit_card, hit_probability in PROBABILITY_DICT.items():
                if x + hit_card <= 21:
                    sum_for_hits += hit_probability * v_prev[x + hit_card, y]
                else:
                    sum_for_hits += hit_probability * -1

            v_f[x, y] = max(r[x, y], sum_for_hits)
            actions[x, y] = np.argmax([r[x, y], sum_for_hits])
            halt_flag = np.array_equal(v_f, v_prev)
            v_prev = np.copy(v_f)

fig = plt.figure()
ax = plt.axes(projection='3d')

y = np.linspace(4, 22, 19)
x = np.linspace(2, 11, 10)

X, Y = np.meshgrid(x, y)
ax.plot_surface(X=X, Y=Y, Z=v_f[4:, 2:])
ax.set_xlabel("Dealer Showing")
ax.set_ylabel('Player Sum')
ax.set_zlabel('Value function')
ax.set_xlim([2, 11])
ax.set_ylim([4, 22])
plt.show()

_, ax = plt.subplots()
ax.imshow(actions[4:, 2:], extent=[2, 11, 21, 4])
ax.set_xlabel("Dealer Showing")
ax.set_ylabel('Player Sum')
plt.show()
