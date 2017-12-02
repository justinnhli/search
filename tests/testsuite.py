#!/usr/bin/env python3

import unittest

from search import algorithms
from search import domains


class Search_GridWorld(unittest.TestCase):

    def setUp(self):
        self.start_state = domains.GridWorldState(0, 0, 20, 20)
        self.goal_state = domains.GridWorldState(19, 15, 20, 20)

    def test_bfs(self):
        final_node = algorithms.GoalOrientedBreadthFirstSearch(goal_state=self.goal_state).search_node(self.start_state)
        self.assertIsNotNone(final_node)
        self.assertEqual(len(final_node.path()), 35)

    def test_ucs(self):
        final_node = algorithms.GoalOrientedUniformCostSearch(goal_state=self.goal_state).search_node(self.start_state)
        self.assertIsNotNone(final_node)
        self.assertEqual(len(final_node.path()), 35)

    def test_astar(self):
        final_node = algorithms.AStarSearch(goal_state=self.goal_state).search_node(self.start_state)
        self.assertIsNotNone(final_node)
        self.assertEqual(len(final_node.path()), 35)


class Search_SlidingPuzzle(unittest.TestCase):

    def setUp(self):
        self.start_state = domains.SlidingPuzzleState('48 653712')
        self.goal_state = domains.SlidingPuzzleState(' 12345678')

    def test_astar(self):
        final_node = algorithms.AStarSearch(goal_state=self.goal_state).search_node(self.start_state)
        self.assertIsNotNone(final_node)
        self.assertEqual(len(final_node.path()), 21)


class Search_Polynomial(unittest.TestCase):

    def test_beam_left(self):
        start_state = domains.PolynomialDescentState(-10)
        goal_state = domains.PolynomialDescentState(-1)
        self.assertEqual(algorithms.HillClimbing().search(start_state), goal_state)

    def test_beam_middle(self):
        start_state = domains.PolynomialDescentState(0)
        goal_state = domains.PolynomialDescentState(-1)
        self.assertEqual(algorithms.HillClimbing().search(start_state), goal_state)

    def test_beam_right(self):
        start_state = domains.PolynomialDescentState(10)
        goal_state = domains.PolynomialDescentState(-1)
        self.assertEqual(algorithms.HillClimbing().search(start_state), goal_state)


if __name__ == "__main__":
    unittest.main()
