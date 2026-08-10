# agent.py
import random


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