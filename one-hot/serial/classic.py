"""Predator-Prey Task (Classical Approach)
This implements the Predator-Prey task within classical computing using the 
serial approach (i.e. at each time step, allocate attention, observe locations,
and decide on optimal movement direction).
"""

import time
from models import attention_classical as attention_mod
from models import movement_classical as movement_mod
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod
from metrics import metrics as metrics_mod

ITERATIONS = 17 # Number of iterations in the game
WIDTH = HEIGHT = 100
MAX_SPEED = 5
        
def main():
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Serial Classical Implementation")

    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)

    # Initialize the attention allocation model
    attention_model = attention_mod.AttentionModelClassical(WIDTH, HEIGHT)

    # Initialize the movement model
    movement_model = movement_mod.MovementModelClassical(WIDTH, HEIGHT)

    # Run model for n iterations
    for _ in range(ITERATIONS):
        # Start the attention timer (must be adjusted from seconds to microseconds)
        attn_start_time = time.time() * 1000000
        # Allocate attention given the real locations of the characters
        attn_agent, attn_prey, attn_predator = attention_model.get_attention_levels(agent,
                                                                                    prey,
                                                                                    predator)
        # Record time elapsed during attention allocation
        metrics.attention_time += (time.time() * 1000000 - attn_start_time)

        # Prey avoids agent
        prey.avoid(agent.loc, MAX_SPEED)
        # Predator pursues agent
        predator.pursue(agent.loc, MAX_SPEED)

        # Get the perceived locations
        agent_perceived = agent.perceive(agent, attn_agent)
        prey_perceived = agent.perceive(prey, attn_prey)
        predator_perceived = agent.perceive(predator, attn_predator)

        # Start the movement timer
        movement_start_time = time.time() * 1000000
        # Pass the perceived locations to the movement model and move to the location
        # determined by it
        movement_model.move(agent, agent_perceived, prey_perceived, predator_perceived,
                            prey.loc, predator.loc, MAX_SPEED)
        # Record time elapsed during movement
        metrics.movement_time += (time.time() * 1000000 - movement_start_time)

    # Add general metrics
    metrics.w = WIDTH
    metrics.h = HEIGHT
    metrics.iterations = ITERATIONS
    metrics.max_speed = MAX_SPEED

    # Add agent to metrics
    metrics.agent_alive = agent.alive
    metrics.agent_feasted = agent.feasted
    metrics.agent_loc_trace = agent.loc_trace
    metrics.agent_perceived_loc_trace = agent.perceived_agent_trace
    metrics.prey_perceived_loc_trace = agent.perceived_prey_trace
    metrics.predator_perceived_loc_trace = agent.perceived_predator_trace
    metrics.dist_agent2prey_trace = [dist[0] for dist in agent.dist_trace]
    metrics.dist_agent2predator_trace = [dist[1] for dist in agent.dist_trace]

    # Add prey to metrics
    metrics.prey_alive = prey.alive
    metrics.prey_loc_trace = prey.loc_trace

    # Add predator to metrics
    metrics.predator_feasted = predator.feasted
    metrics.predator_loc_trace = predator.loc_trace

    # Add attention trace to metrics
    metrics.attention_trace = agent.attn_trace

    # Display metrics
    print(metrics)

    return

if __name__ == "__main__":
    main()
