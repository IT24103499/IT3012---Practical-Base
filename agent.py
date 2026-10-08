# agent.py
import heapq
import math
import random
from collections import deque

from logic_engine import KnowledgeBase


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']
        self.heading = 'Right'  # The agent only knows which way it last moved

    def sense_and_act(self, percept: dict) -> str:
        # The percept is purely local now: no coordinates to steer by.
        if percept['food_here']:
            return 'Suck'
        if percept['wall_ahead']:
            # Blocked in front, so sweep off in some other direction
            self.heading = random.choice([a for a in self.actions_pool if a != self.heading])
        return self.heading


class SimpleReflexAgent:
    """A purely reactive agent: it maps the current percept straight to an action.

    There is deliberately no __init__ and no internal state. The agent has no
    memory of where it has been, so identical percepts always produce identical
    actions -- which is exactly what makes it loop forever in a dead end.
    """

    def sense_and_act(self, percept: dict) -> str:
        # Condition-Action rules, applied strictly in order
        if percept['food_here']:        # IF food here
            return 'Suck'               # THEN eat it
        if percept['wall_ahead']:       # IF blocked in front
            return 'TurnLeft'           # THEN turn left
        return 'Forward'                # ELSE keep going


class ModelBasedAgent:
    """A reflex agent with an internal model of the world.

    The percept contains no coordinates, so the agent tracks its own position
    by dead reckoning: it knows which action it issued last, so it knows how
    the world must have changed. Positions are relative to wherever it started,
    which is all it needs to answer "have I been here before?".
    """

    TURN_ORDER = ['Right', 'Up', 'Left', 'Down']  # Counter-clockwise
    DELTAS = {'Right': (1, 0), 'Up': (0, 1), 'Left': (-1, 0), 'Down': (0, -1)}

    def __init__(self):
        # --- Internal state (the "model") ---
        self.pos = (0, 0)          # Position relative to the starting square
        self.heading = 'Right'     # Assumed starting facing, matches the environment
        self.visited_cells = {(0, 0)}
        self.known_walls = set()   # Cells discovered to be blocked
        self.last_action = None

    def _cell_towards(self, direction):
        dx, dy = self.DELTAS[direction]
        return (self.pos[0] + dx, self.pos[1] + dy)

    def sense_and_act(self, percept: dict) -> str:
        # --- 1. TRANSITION MODEL: how did my last action change the world? ---
        if self.last_action == 'Forward':
            self.pos = self._cell_towards(self.heading)
        elif self.last_action in ('TurnLeft', 'TurnRight'):
            i = self.TURN_ORDER.index(self.heading)
            step = 1 if self.last_action == 'TurnLeft' else -1
            self.heading = self.TURN_ORDER[(i + step) % len(self.TURN_ORDER)]

        # --- 2. SENSOR MODEL: record what I can see from here ---
        self.visited_cells.add(self.pos)
        ahead = self._cell_towards(self.heading)
        if percept['wall_ahead']:
            self.known_walls.add(ahead)
        else:
            self.known_walls.discard(ahead)

        # --- 3. Condition-Action rules, now querying memory ---
        action = self._choose_action(percept)
        self.last_action = action
        return action

    def _choose_action(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Suck'

        # Keep going only if the cell in front is open AND somewhere new
        ahead = self._cell_towards(self.heading)
        if not percept['wall_ahead'] and ahead not in self.visited_cells:
            return 'Forward'

        # Otherwise consult the map: which neighbours are not known walls?
        open_dirs = [d for d in self.TURN_ORDER if self._cell_towards(d) not in self.known_walls]
        # Prefer somewhere we have never been -- this is what breaks the loop
        unexplored = [d for d in open_dirs if self._cell_towards(d) not in self.visited_cells]
        candidates = unexplored or open_dirs

        if not candidates:
            return 'TurnLeft'  # Fully boxed in, keep scanning

        target = self._nearest_by_rotation(candidates)
        if target == self.heading:
            return 'Forward'

        # e.g. IF wall_ahead AND the left cell is visited/blocked THEN turn right
        i, j = self.TURN_ORDER.index(self.heading), self.TURN_ORDER.index(target)
        return 'TurnLeft' if (j - i) % 4 <= (i - j) % 4 else 'TurnRight'

    def _nearest_by_rotation(self, candidates):
        """Pick the candidate direction needing the fewest turns to face."""
        i = self.TURN_ORDER.index(self.heading)

        def turns(d):
            j = self.TURN_ORDER.index(d)
            return min((j - i) % 4, (i - j) % 4)

        return min(candidates, key=turns)


class SearchAgent:
    """A goal-based (planning) agent.

    Instead of reacting to the cell in front of it, this agent receives the
    global state space in the percept ('grid_size', 'walls', 'all_food') and
    *simulates* journeys through it before moving. Each search below is a
    GRAPH search: the `reached` set records every state that has already been
    expanded, so the agent never re-explores a cell and never falls into the
    infinite loop a plain tree search would.
    """

    DELTAS = {'Right': (1, 0), 'Up': (0, 1), 'Left': (-1, 0), 'Down': (0, -1)}

    def __init__(self, active_algo='BFS'):
        self.plan = []                    # Queued actions still to be executed
        self.active_algo = active_algo    # Which search to plan with: 'BFS', 'DFS' or 'UCS'
        self.pos = (0, 0)                 # Dead-reckoned position (the percept has no coordinates)

        # Knowledge Base of safety rules (Horn clauses) consulted by A*
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')          # Rule 1
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')     # Rule 2

    # ------------------------------------------------------------------
    # Agent program: plan once, then execute the plan one action per tick
    # ------------------------------------------------------------------
    def sense_and_act(self, percept: dict) -> str:
        # Standing on a pellet is always worth a free Suck
        if percept['food_here']:
            return 'Suck'

        # Per-tile facts the game does not produce itself, supplied as {(x, y): [facts]}
        tile_facts = {
            (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],  # deduces Retreat: blocked
            (0, 1): ['TargetVisible', 'HasDust'],                        # SafeToEngage only: allowed
        }

        if not self.plan:                 # No plan in memory -> think before moving
            self.plan = self._formulate_plan(percept, tile_facts)

        if not self.plan:                 # Nothing reachable: rotate and re-sense
            return 'TurnLeft'

        action = self.plan.pop(0)         # Execute the next step of the plan
        if action in self.DELTAS:         # Keep the internal position model in sync
            dx, dy = self.DELTAS[action]
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        return action

    def _formulate_plan(self, percept: dict, tile_facts=None) -> list:
        """GOAL FORMULATION + SEARCH: pick the closest pellet, then simulate a
        route to it with the configured algorithm and translate it to actions."""
        food = [tuple(f) for f in percept['all_food']]
        if not food:
            return []

        walls = percept['walls']
        grid_size = percept['grid_size']

        # Closest by Manhattan distance -- the goal we commit to searching for
        goal = min(food, key=lambda f: abs(f[0] - self.pos[0]) + abs(f[1] - self.pos[1]))

        algo = self.active_algo.upper()
        if algo == 'BFS':
            path = self.bfs_search(self.pos, goal, walls, grid_size)
        elif algo == 'DFS':
            path = self.dfs_search(self.pos, goal, walls, grid_size)
        elif algo == 'UCS':
            path, _cost = self.ucs_search(self.pos, goal, walls, grid_size)
        elif algo == 'ASTAR':
            path = self.astar_search(self.pos, goal, walls, grid_size, 'manhattan', tile_facts=tile_facts)
        else:
            raise ValueError(f"Unknown search algorithm: {self.active_algo}")

        if path is None:
            # That pellet is walled off; retry with every pellet as a goal,
            # treating logically infeasible tiles as walls too
            blocked = set(map(tuple, walls))
            blocked |= {cell for cell in (tile_facts or {}) if not self.is_feasible(cell, tile_facts)}
            path = self.bfs_search(self.pos, food, blocked, grid_size)
            if path is None:
                return []

        return self._path_to_actions(path)

    def _path_to_actions(self, path: list) -> list:
        """Turn a list of cells into the movement actions that walk it,
        finishing with a Suck on the goal square."""
        actions = []
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            step = (x2 - x1, y2 - y1)
            for name, delta in self.DELTAS.items():
                if delta == step:
                    actions.append(name)
                    break
        actions.append('Suck')
        return actions

    # ------------------------------------------------------------------
    # Successor function (the transition model the searches expand over)
    # ------------------------------------------------------------------
    def _successors(self, state, walls, grid_size):
        """Return every legal neighbouring cell of `state`."""
        width, height = grid_size
        x, y = state
        result = []
        for dx, dy in self.DELTAS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                result.append((nx, ny))
        return result

    @staticmethod
    def _goal_set(goals):
        """Accept a single (x, y) goal or a collection of them."""
        if isinstance(goals, tuple) and len(goals) == 2 and all(isinstance(c, int) for c in goals):
            return {goals}
        return set(goals)

    # ------------------------------------------------------------------
    # Feasibility check -- ask the Knowledge Base whether a tile is safe
    # ------------------------------------------------------------------
    def is_feasible(self, cell, tile_facts):
        """Load the tile's facts into the KB, forward chain, and report the
        tile infeasible if 'Retreat' is deduced."""
        self.kb.clear_facts()
        for fact in tile_facts.get(cell, []):
            self.kb.tell_fact(fact)
        self.kb.forward_chain()
        return 'Retreat' not in self.kb.facts

    # ------------------------------------------------------------------
    # Heuristic functions h(n) -- estimated cost from a cell to the goal
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|, the grid distance with no diagonals."""
        return int(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2), the straight-line distance."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # ------------------------------------------------------------------
    # 1. Breadth-First Search -- FIFO queue, shallowest node first
    # ------------------------------------------------------------------
    def bfs_search(self, start, goals, walls, grid_size):
        """Shortest path in number of steps. Returns [start, ..., goal] or None."""
        start = tuple(start)
        goals = self._goal_set(goals)
        walls = set(map(tuple, walls))

        if start in goals:
            return [start]

        frontier = deque([(start, [start])])   # FIFO
        reached = {start}                      # Graph search: never revisit a state

        while frontier:
            state, path = frontier.popleft()   # <-- shallowest node expands first

            for nxt in self._successors(state, walls, grid_size):
                if nxt in reached:
                    continue
                if nxt in goals:               # Early goal test: BFS is optimal here
                    return path + [nxt]
                reached.add(nxt)
                frontier.append((nxt, path + [nxt]))

        return None  # Goal unreachable

    # ------------------------------------------------------------------
    # 2. Depth-First Search -- LIFO stack, deepest node first
    # ------------------------------------------------------------------
    def dfs_search(self, start, goals, walls, grid_size):
        """Dives down one branch as far as possible. Complete (graph search)
        but NOT optimal: the path it finds is usually far from the shortest."""
        start = tuple(start)
        goals = self._goal_set(goals)
        walls = set(map(tuple, walls))

        frontier = [(start, [start])]  # LIFO
        reached = set()                # Graph search: guards against cycles

        while frontier:
            state, path = frontier.pop()   # <-- deepest node expands first

            if state in reached:
                continue
            reached.add(state)

            if state in goals:            # Late goal test: no optimality to protect
                return path

            for nxt in self._successors(state, walls, grid_size):
                if nxt not in reached:
                    frontier.append((nxt, path + [nxt]))

        return None  # Goal unreachable

    # ------------------------------------------------------------------
    # 3. Uniform-Cost Search -- priority queue ordered by path cost g(n)
    # ------------------------------------------------------------------
    def ucs_search(self, start, goals, walls, grid_size, cost_fn=None):
        """Cheapest-first expansion. `cost_fn(cell)` gives the cost of entering
        a cell (default 1 everywhere, which degenerates to BFS). Returns
        (path, total_cost), or (None, inf) if no goal is reachable."""
        start = tuple(start)
        goals = self._goal_set(goals)
        walls = set(map(tuple, walls))
        if cost_fn is None:
            cost_fn = lambda cell: 1

        counter = 0  # Tie-breaker so heapq never has to compare the path lists
        frontier = [(0, counter, start, [start])]
        best_cost = {start: 0}   # Cheapest g(n) found so far per state
        reached = set()          # States already expanded

        while frontier:
            g, _, state, path = heapq.heappop(frontier)  # <-- cheapest node first

            if state in reached:
                continue
            reached.add(state)

            if state in goals:   # Late goal test: required for UCS optimality
                return path, g

            for nxt in self._successors(state, walls, grid_size):
                if nxt in reached:
                    continue
                new_g = g + cost_fn(nxt)
                if new_g < best_cost.get(nxt, float('inf')):
                    best_cost[nxt] = new_g
                    counter += 1
                    heapq.heappush(frontier, (new_g, counter, nxt, path + [nxt]))

        return None, float('inf')

    # ------------------------------------------------------------------
    # 4. A* Search -- priority queue ordered by f(n) = g(n) + h(n)
    # ------------------------------------------------------------------
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan', tile_facts=None):
        """Informed search: expands the node with the lowest estimated total
        cost f(n) = g(n) + h(n). Tiles where the KB deduces 'Retreat' are
        skipped as infeasible. Returns [start, ..., goal] or None."""
        if tile_facts is None:
            tile_facts = {}
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        walls = set(map(tuple, walls))

        # Pick the heuristic h(n) to guide the search
        if heuristic_type == 'manhattan':
            heuristic = self.manhattan_distance
        elif heuristic_type == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_type}")

        frontier = []           # Priority queue ordered by f_cost
        reached_states = set()  # Graph search: states already expanded

        # Initial node: g(n) = 0, so f(n) = h(n)
        g_cost = 0
        h_cost = heuristic(start_pos, goal_pos)
        heapq.heappush(frontier, (g_cost + h_cost, g_cost, start_pos, [start_pos]))

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:   # Goal test on expansion keeps A* optimal
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            # Expand the four adjacent cells (Up, Down, Left, Right).
            # _successors() checks REACHABILITY (in bounds, not a wall)...
            for neighbour in self._successors(current_pos, walls, grid_size):
                if neighbour in reached_states:
                    continue
                # ...and the KB checks FEASIBILITY: skip tiles where Retreat is deduced
                if not self.is_feasible(neighbour, tile_facts):
                    continue
                g_new = g_cost + 1
                h_new = heuristic(neighbour, goal_pos)
                f_new = g_new + h_new
                heapq.heappush(frontier, (f_new, g_new, neighbour, path_taken + [neighbour]))

        return None  # Goal unreachable
