from a_star import solve as solve_a_star
from dijkstra import solve as solve_dijkstra
from Wet_1.dijkstra import *
import random
import pickle
import sys, os


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def createPuzzles(n_tests):
    puzzles = []
    n_actions = []
    for i in range(n_tests):
        initial_state = State()
        n = random.randint(0, 100)
        actions = []
        goal_state = initial_state
        for a in range(n):
            actions.append(random.choice(goal_state.get_actions()))
            goal_state = goal_state.apply_action(actions[a])
        puzzles.append(Puzzle(initial_state, goal_state))
        n_actions.append(n)
    with open('puzzles.pkl', 'wb') as f:
        pickle.dump([puzzles, n_actions], f)


def run_tests():
    with open('puzzles.pkl', 'rb') as f:
        puzzles, n_actions = pickle.load(f)
    plan_lengths_dijkstra = []
    plan_lengths_a_star = []
    for i in range(len(puzzles)):
        blockPrint()
        plan_dijkstra = solve_dijkstra(puzzles[i])
        plan_a_star = solve_a_star(puzzles[i])
        enablePrint()
        if len(plan_dijkstra) - 1 > n_actions[i]:
            print("Wrong solution with dijkstra for test " + str(i))
        if len(plan_a_star) - 1 > n_actions[i]:
            print("Wrong solution with a_star for test " + str(i))
        plan_lengths_dijkstra.append(len(plan_dijkstra) - 1)
        plan_lengths_a_star.append(len(plan_a_star) - 1)
        print("ran test num " + str(i))
    with open('results.pkl', 'wb') as f:
        pickle.dump([plan_lengths_dijkstra, plan_lengths_a_star], f)


def compare_tests():
    with open('results_comp.pkl', 'rb') as f:
        plan_lengths_comp, _ = pickle.load(f)
    with open('results.pkl', 'rb') as f:
        plan_lengths_dijkstra, plan_lengths_a_star = pickle.load(f)
    for i in range(len(plan_lengths_comp)):
        if plan_lengths_dijkstra[i] != plan_lengths_comp[i]:
            print("mismatch for dijkstra in test " + str(i))
        elif plan_lengths_a_star[i] != plan_lengths_comp[i]:
            print("mismatch for a_star in test " + str(i))
        else:
            print("match in test " + str(i))


run_tests()
compare_tests()

# with open('puzzles.pkl', 'rb') as f:
#     puzzles, n_actions = pickle.load(f)
# plan_dijkstra = solve_dijkstra(puzzles[30])
# plan_a_star = solve_a_star(puzzles[30])
# print('length of plan dijkstra: ' + str(len(plan_dijkstra) - 1))
# print('length of plan a_star: ' + str(len(plan_a_star) - 1))
# print('length of actions: ' + str(n_actions[30]))
#
# plan_dijkstra = solve_dijkstra(puzzles[84])
# plan_a_star = solve_a_star(puzzles[84])
# print('length of plan dijkstra: ' + str(len(plan_dijkstra) - 1))
# print('length of plan a_star: ' + str(len(plan_a_star) - 1))
# print('length of actions: ' + str(n_actions[84]))
