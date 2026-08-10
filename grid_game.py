# grid_game.py
import random


class GridHuntGame:
    """A small Pacman-style grid environment (4x4) where an agent collects food."""

    # Offset applied to (x, y) for each movement action / facing direction
    DIRECTIONS = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }

    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.agent_dir = 'Right'  # Direction the agent is currently facing

        # Place a few random food pellets and obstacles (walls)
        self.food_positions = {(1, 2), (2, 3), (3, 0), (2, 1)}
        self.walls = {(1, 1), (2, 2)}

        self.score = 0
        self.steps = 0

    def get_percept(self, agent) -> dict:
        return {
            'agent_pos': list(self.agent_pos),
            'smells_food': tuple(self.agent_pos) in self.food_positions,
            'hit_wall': tuple(self.agent_pos) in self.walls,
            'score': self.score,
            'remaining_food': len(self.food_positions)
        }

    # Counter-clockwise / clockwise rotation order used by TurnLeft and TurnRight
    TURN_ORDER = ['Right', 'Up', 'Left', 'Down']

    def execute_action(self, agent, action: str):
        self.steps += 1
        new_pos = list(self.agent_pos)

        if action == 'Suck':
            # Consume the pellet underfoot, if there is one
            tuple_pos = tuple(self.agent_pos)
            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20  # Reward for eating food pellet
            return

        if action in ('TurnLeft', 'TurnRight'):
            # Rotating in place: the agent changes facing but does not move
            i = self.TURN_ORDER.index(self.agent_dir)
            step = 1 if action == 'TurnLeft' else -1
            self.agent_dir = self.TURN_ORDER[(i + step) % len(self.TURN_ORDER)]
            return

        if action == 'Forward':
            dx, dy = self.DIRECTIONS[self.agent_dir]
            new_pos[0] += dx
            new_pos[1] += dy
        elif action in self.DIRECTIONS:
            self.agent_dir = action  # An absolute move also turns the agent that way
            dx, dy = self.DIRECTIONS[action]
            new_pos[0] += dx
            new_pos[1] += dy

        # Clamp to the grid: walking off the edge leaves the agent where it was
        new_pos[0] = max(0, min(self.width - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.height - 1, new_pos[1]))

        # Check collision with walls
        if tuple(new_pos) in self.walls:
            self.score -= 5  # Penalty for hitting a wall
        else:
            self.agent_pos = new_pos

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 20