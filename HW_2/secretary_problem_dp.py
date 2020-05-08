import matplotlib.pyplot as plt

N = 10
value_function_0 = [-1 for _ in range(N)]
value_function_0[-1] = 0
value_function_1 = [-1 for _ in range(N)]
value_function_1[-1] = 1
g_t_1 = lambda t: t / N

for iter in range(1, N):
    t_step: int = N - iter - 1
    p_1 = 1 / (t_step + 1)
    p_0 = t_step / (t_step + 1)
    value_function_0[t_step] = p_1 * value_function_1[t_step + 1] + p_0 * value_function_0[t_step + 1]
    value_function_1[t_step] = max([g_t_1(t_step), value_function_0[t_step]])

# Plot the data
plt.plot(range(0, N), value_function_0,label="Value function at states 0")

# Add a titles

# Plot the data
plt.plot(range(0, N), value_function_1,label="Value function at states 1")
plt.legend()

# Add a titles
plt.xlabel("Time")
plt.ylabel("Value Function at state")
plt.title("Value Function Vs. Time")
plt.show()