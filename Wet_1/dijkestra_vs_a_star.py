from puzzle import *
from dijkstra import solve as dijkstra_solver
from a_star import solve as a_star_solver
import datetime

if __name__ == "__main__":
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State("8 7 6\r\n1 0 5\r\n2 3 4")
    actions = ['d', 'l', 'u', 'r', 'r', 'd', 'l', 'u', 'u', 'r', 'd', 'd', 'l', 'u', 'u', 'l', 'd', 'd', 'r', 'u', 'u', 'l', 'd', 'r', 'r']
    goal_state = State("1 3 4\r\n8 2 0\r\n7 6 5")
    puzzle = Puzzle(initial_state, goal_state)
    solution_start_time = datetime.datetime.now()
    dijkstra_solver(puzzle)
    print('dijkstra_solver time to solve {}'.format(datetime.datetime.now() - solution_start_time))

    solution_start_time = datetime.datetime.now()
    a_star_solver(puzzle)
    print('a_star_solver time to solve {}'.format(datetime.datetime.now() - solution_start_time))
