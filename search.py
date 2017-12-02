from functools import partial


class State:

    '''A generic problem state
    '''

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def __repr__(self):
        args = ['{}={}'.format(k, repr(v)) for k, v in sorted(vars(self).items()) if not k.startswith('_')]
        return '{}({})'.format(type(self).__name__, ', '.join(args))

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def to_tuple(self):
        return tuple([v for k, v in sorted(vars(self).items())])

    def actions(self):
        raise NotImplementedError

    def heuristic(self, goal=None):
        return 0

    def random_state(self):
        return self


class Action:

    '''A generic action
    '''

    def __init__(self, name, next_state, cost):
        self.name = name
        self.next_state = next_state
        self.cost = cost

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Action({})'.format(repr(self.name))


class PathNode:

    '''A wrapper around the state that stores contextual information

    This class is a linked list, where each node points to the previous node
    and the action that led from the previous node to this node.
    '''

    def __init__(self, state, prev_node=None, prev_edge=None):
        self.state = state
        self.prev_node = prev_node
        self.prev_edge = prev_edge
        self._total_cost = None

    def __len__(self):
        return len(list(iter(self)))

    def __iter__(self):
        node = self
        while node is not None:
            yield node
            node = node.prev_node

    def __str__(self):
        return 'PathNode({})'.format(str(self.state))

    def __repr__(self):
        return 'PathNode({})'.format(repr(self.state))

    def path(self):
        return list(iter(self))

    def total_cost(self):
        if self._total_cost is None:
            self._total_cost = sum(node.prev_edge.cost for node in iter(self) if node.prev_edge)
        return self._total_cost


class AbstractSearch:

    '''
    termination_fn: [state] -> bool
        given a list of states in the queue, return whether the search should terminate
    batch_fn: [node] -> [[node], [node]]
        given a list of states in the queue, determine which states should be used in the next batch and which should be in saved for later
    population_fn:  [[node], [node]] -> [node]
        given a list of states saved for later and a list of newly generated states, determine the states to be in the queue
    '''

    def __init__(self, termination_fn, batch_fn, population_fn, memoryless=True):
        # search parameters
        self.termination_fn = termination_fn
        self.batch_fn = batch_fn
        self.population_fn = population_fn
        self.memoryless = memoryless
        # search variables
        self.population = []
        self.visited = set()

    def expand_batch(self, batch):
        frontier = []
        for cur_node in batch:
            cur_state = cur_node.state
            if not self.memoryless:
                if cur_state in self.visited:
                    continue
                self.visited.add(cur_state)
            for action in cur_state.actions():
                if action.next_state not in self.visited:
                    if self.memoryless:
                        next_node = PathNode(action.next_state, None, action)
                    else:
                        next_node = PathNode(action.next_state, cur_node, action)
                    frontier.append(next_node)
        return frontier

    def search_nodes(self, start_state):
        cur_generation = [PathNode(start_state)]
        self.visited = set()
        while not self.termination_fn([node.state for node in cur_generation]):
            next_generation = []
            batch, remainder = self.batch_fn(cur_generation)
            next_generation = self.expand_batch(batch)
            cur_generation = self.population_fn(remainder, next_generation)
        return cur_generation

    def search_node(self, start_state):
        nodes = self.search_nodes(start_state)
        if nodes:
            return nodes[0]
        else:
            return None

    def search_state(self, start_state):
        node = self.search_node(start_state)
        if node:
            return node.state
        else:
            return None

    def search(self, start_state):
        return self.search_state(start_state)

    def random_restart_search(self, start_state, num_restarts=100):
        best_state = None
        best_heuristic = None
        for _ in range(num_restarts):
            final_state = self.search(start_state.random_state())
            state_heuristic = final_state.heuristic()
            if best_state is None or state_heuristic < best_heuristic:
                best_state = final_state
                best_heuristic = state_heuristic
        return best_state


class DepthFirstSearch(AbstractSearch):

    @staticmethod
    def _batch_fn(cur_generation):
        generation = sorted(cur_generation, key=(lambda node: node.total_cost()), reverse=True)
        return [generation[0]], generation[1:]

    @staticmethod
    def _population_fn(remainder, next_generation):
        result = sorted(
            remainder + next_generation,
            key=(lambda node: (node.total_cost(), *node.state.to_tuple())),
            reverse=True,
        )
        return result

    def __init__(self, termination_fn):
        super().__init__(
            termination_fn,
            batch_fn=DepthFirstSearch._batch_fn,
            population_fn=DepthFirstSearch._population_fn,
            memoryless=False,
        )


class GoalOrientedDepthFirstSearch(DepthFirstSearch):

    @staticmethod
    def _termination_fn(state_list, goal_state):
        return state_list and state_list[0] == goal_state

    def __init__(self, goal_state):
        self.goal_state = goal_state
        termination_fn = (lambda state_list: GoalOrientedDepthFirstSearch._termination_fn(state_list, self.goal_state))
        super().__init__(termination_fn=termination_fn)


class BreadthFirstSearch(AbstractSearch):

    @staticmethod
    def _batch_fn(cur_generation):
        generation = sorted(cur_generation, key=(lambda node: node.total_cost()))
        return [generation[0]], generation[1:]

    @staticmethod
    def _population_fn(remainder, next_generation):
        result = sorted(
            remainder + next_generation,
            key=(lambda node: (node.total_cost(), *node.state.to_tuple())),
        )
        return result

    def __init__(self, termination_fn):
        super().__init__(
            termination_fn,
            batch_fn=BreadthFirstSearch._batch_fn,
            population_fn=BreadthFirstSearch._population_fn,
            memoryless=False,
        )


class GoalOrientedBreadthFirstSearch(BreadthFirstSearch):

    @staticmethod
    def _termination_fn(state_list, goal_state):
        return state_list and state_list[0] == goal_state

    def __init__(self, goal_state):
        self.goal_state = goal_state
        termination_fn = (
            lambda state_list: GoalOrientedBreadthFirstSearch._termination_fn(state_list, self.goal_state)
        )
        super().__init__(termination_fn=termination_fn)


class UniformCostSearch(AbstractSearch):

    @staticmethod
    def _termination_fn(state_list, goal_state):
        return state_list and state_list[0] == goal_state

    @staticmethod
    def _batch_fn(cur_generation):
        generation = sorted(cur_generation, key=(lambda node: node.total_cost()))
        return [generation[0]], generation[1:]

    @staticmethod
    def _population_fn(remainder, next_generation):
        result = sorted(
            remainder + next_generation,
            key=(lambda node: (node.total_cost(), *node.state.to_tuple())),
        )
        return result

    def __init__(self, goal_state):
        self.goal_state = goal_state
        termination_fn = (lambda state_list: GreedyBestFirstSearch._termination_fn(state_list, self.goal_state))
        super().__init__(
            termination_fn=termination_fn,
            batch_fn=UniformCostSearch._batch_fn,
            population_fn=UniformCostSearch._population_fn,
            memoryless=False,
        )


class GreedyBestFirstSearch(AbstractSearch):

    @staticmethod
    def _termination_fn(state_list, goal_state):
        return state_list and state_list[0] == goal_state

    @staticmethod
    def _batch_fn(cur_generation):
        generation = sorted(cur_generation, key=(lambda node: node.state.heuristic()))
        return [generation[0]], generation[1:]

    @staticmethod
    def _population_fn(remainder, next_generation):
        result = sorted(
            remainder + next_generation,
            key=(lambda node: (node.state.heuristic(), *node.state.to_tuple())),
        )
        return result

    def __init__(self, goal_state):
        self.goal_state = goal_state
        termination_fn = (lambda state_list: GreedyBestFirstSearch._termination_fn(state_list, self.goal_state))
        super().__init__(
            termination_fn=termination_fn,
            batch_fn=GreedyBestFirstSearch._batch_fn,
            population_fn=GreedyBestFirstSearch._population_fn,
            memoryless=False,
        )


class AStarSearch(AbstractSearch):

    @staticmethod
    def _termination_fn(state_list, goal_state):
        return state_list and state_list[0] == goal_state

    @staticmethod
    def _batch_fn(cur_generation, goal_state):
        generation = sorted(
            cur_generation,
            key=(lambda node: node.total_cost() + node.state.heuristic(goal_state)),
        )
        return [generation[0]], generation[1:]

    @staticmethod
    def _population_fn(remainder, next_generation, goal_state):
        result = sorted(
            remainder + next_generation,
            key=(lambda node: (node.total_cost() + node.state.heuristic(goal_state), *node.state.to_tuple())),
        )
        return result

    def __init__(self, goal_state):
        self.goal_state = goal_state
        termination_fn = (lambda state_list: AStarSearch._termination_fn(state_list, self.goal_state))
        super().__init__(
            termination_fn=termination_fn,
            batch_fn=(lambda cur_generation: AStarSearch._batch_fn(cur_generation, self.goal_state)),
            population_fn=(lambda remainder, next_generation: AStarSearch._population_fn(remainder, next_generation, self.goal_state)),
            memoryless=False,
        )


class HillClimbing(AbstractSearch):

    @staticmethod
    def _termination_fn(state_list):
        curr_state = state_list[0]
        curr_heuristic = curr_state.heuristic()
        next_heuristics = [action.next_state.heuristic() for action in curr_state.actions()]
        return not any(next_heuristic < curr_heuristic for next_heuristic in next_heuristics)

    @staticmethod
    def _batch_fn(cur_generation):
        generation = sorted(
            cur_generation,
            key=(lambda node: node.state.heuristic()),
        )
        return [generation[0]], generation[1:]

    @staticmethod
    def _population_fn(remainder, next_generation):
        result = min(
            remainder + next_generation,
            key=(lambda node: (node.state.heuristic(), *node.state.to_tuple())),
        )
        return [result]

    def __init__(self):
        super().__init__(
            termination_fn=HillClimbing._termination_fn,
            batch_fn=HillClimbing._batch_fn,
            population_fn=HillClimbing._population_fn,
            memoryless=True,
        )
