from puzzle import *
from a_star import solve as a_star_solver
import pytest


def analyze_plan(plan):
    actions = list()
    for current_state, action in plan:
        if action is not None:
            actions.append(action)

    return {"length": len(plan) - 1, "actions": " ".join(actions)}


def test_6_moves():
    init_state = State("1 3 4\r\n8 0 5\r\n7 2 6")
    goal_state = State("1 2 3\r\n8 0 4\r\n7 6 5")
    puzzle = Puzzle(init_state, goal_state)
    plan = a_star_solver(puzzle)
    plan = analyze_plan(plan)
    assert plan["length"] == 6
    assert plan["actions"] == 'd r u u l d'


def test_14_moves():
    init_state = State("2 3 1\r\n7 0 8\r\n6 5 4")
    goal_state = State("1 2 3\r\n8 0 4\r\n7 6 5")
    puzzle = Puzzle(init_state, goal_state)
    plan = a_star_solver(puzzle)
    plan = analyze_plan(plan)
    assert plan["length"] == 14
    assert plan["actions"] == 'r u l d r d l l u r u l d r'


def test_16_moves():
    init_state = State("2 3 1\r\n8 0 4\r\n7 6 5")
    goal_state = State("1 2 3\r\n8 0 4\r\n7 6 5")
    puzzle = Puzzle(init_state, goal_state)
    plan = a_star_solver(puzzle)
    plan = analyze_plan(plan)
    assert plan["length"] == 16
    assert plan["actions"] == 'l u r r d l u l d r r u l l d r'


def test_16_2_moves():
    init_state = State("1 2 3\r\n8 0 4\r\n7 6 5")
    goal_state = State("2 3 1\r\n8 0 4\r\n7 6 5")
    puzzle = Puzzle(init_state, goal_state)
    plan = a_star_solver(puzzle)
    plan = analyze_plan(plan)
    assert plan["length"] == 16
    assert plan["actions"] == 'l u r r d l l u r d r u l l d r'


def test_28_moves():
    init_state = State("8 7 6\r\n1 0 5\r\n2 3 4")
    goal_state = State("1 2 3\r\n8 0 4\r\n7 6 5")
    puzzle = Puzzle(init_state, goal_state)
    plan = a_star_solver(puzzle)
    plan = analyze_plan(plan)
    assert plan["length"] == 28
    assert plan["actions"] == 'u r d l l u r d d l u u r d d r u u l d d r u l l d r u'
