# simulator.py
from grid_game import GridHuntGame
from agent import SimpleReflexAgent

def run_grid_hunt():
    env = GridHuntGame()
    agent = SimpleReflexAgent()

    print("=== UC Berkeley Style Small Grid Hunt Started ===")
    while not env.is_done():
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(agent, action)
        # The percept is local-only, so the log reads the global state from the environment
        print(f"Percept: {percept} | Action: {action:<9} | Pos: {env.agent_pos} "
              f"Facing: {env.agent_dir:<5} | Food Left: {len(env.food_positions)} | Score: {env.score}")

    print(f"\nGame Over! Final Score: {env.score} after {env.steps} steps.")

if __name__ == "__main__":
    run_grid_hunt()