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

    def __init__(self, memoryless=True):
        # search parameters
        self.memoryless = memoryless
        # search variables
        self.population = []
        self.visited = set()

    def termination_fn(self, state_list):
        raise NotImplementedError()

    def batch_fn(self, curr_generation):
        raise NotImplementedError()

    def population_fn(self, remainder, next_generation):
        raise NotImplementedError()

    def expand_batch(self, batch):
        frontier = []
        for curr_node in batch:
            curr_state = curr_node.state
            if not self.memoryless:
                if curr_state in self.visited:
                    continue
                self.visited.add(curr_state)
            for action in curr_state.actions():
                if action.next_state not in self.visited:
                    if self.memoryless:
                        next_node = PathNode(action.next_state, None, action)
                    else:
                        next_node = PathNode(action.next_state, curr_node, action)
                    frontier.append(next_node)
        return frontier

    def search_nodes(self, start_state):
        curr_generation = [PathNode(start_state)]
        self.visited = set()
        while not self.termination_fn([node.state for node in curr_generation]):
            next_generation = []
            batch, remainder = self.batch_fn(curr_generation)
            next_generation = self.expand_batch(batch)
            curr_generation = self.population_fn(remainder, next_generation)
        return curr_generation

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


class SingleStateSearch(AbstractSearch):

    def batch_fn(self, curr_generation):
        generation = sorted(curr_generation, key=self.priority_fn)
        return [generation[0]], generation[1:]

    def population_fn(self, remainder, next_generation):
        result = sorted(
            remainder + next_generation,
            key=(lambda node: (self.priority_fn(node), *node.state.to_tuple())),
        )
        return result

    def priority_fn(self, node):
        raise NotImplementedError()


class GoalOrientedSearch(AbstractSearch):

    def termination_fn(self, state_list):
        return state_list and state_list[0] == self.goal_state

    def __init__(self, goal_state, **kwargs):
        super().__init__(**kwargs)
        self.goal_state = goal_state


class DepthFirstSearch(SingleStateSearch):

    def priority_fn(self, node):
        return -len(node.path())

    def __init__(self, **kwargs):
        super().__init__(memoryless=False, **kwargs)


class GoalOrientedDepthFirstSearch(DepthFirstSearch, GoalOrientedSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BreadthFirstSearch(SingleStateSearch):

    def priority_fn(self, node):
        return len(node.path())

    def __init__(self, **kwargs):
        super().__init__(memoryless=False, **kwargs)


class GoalOrientedBreadthFirstSearch(BreadthFirstSearch, GoalOrientedSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UniformCostSearch(SingleStateSearch):

    def priority_fn(self, node):
        return node.total_cost()

    def __init__(self, **kwargs):
        super().__init__(memoryless=False, **kwargs)


class GoalOrientedUniformCostSearch(UniformCostSearch, GoalOrientedSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GreedyBestFirstSearch(SingleStateSearch, GoalOrientedSearch):

    def priority_fn(self, node):
        return node.state.heuristic(self.goal_state)

    def __init__(self, **kwargs):
        super().__init__(memoryless=False, **kwargs)


class AStarSearch(SingleStateSearch, GoalOrientedSearch):

    def priority_fn(self, node):
        return node.total_cost() + node.state.heuristic(self.goal_state)

    def __init__(self, **kwargs):
        super().__init__(memoryless=False, **kwargs)


class HillClimbing(SingleStateSearch):

    def termination_fn(self, state_list):
        curr_state = state_list[0]
        curr_heuristic = curr_state.heuristic()
        next_heuristics = [action.next_state.heuristic() for action in curr_state.actions()]
        return not any(next_heuristic < curr_heuristic for next_heuristic in next_heuristics)

    def priority_fn(self, node):
        return node.state.heuristic()

    def __init__(self, **kwargs):
        super().__init__(memoryless=True, **kwargs)
