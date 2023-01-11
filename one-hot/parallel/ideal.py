""" Complete Predator-Prey Task (Classical Approach)
This implements the complete Predator-Prey task within classical computing using a
restricted boltzmann machine.
"""

import time
import math
import numpy as np
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod
from metrics import metrics as metrics_mod

WIDTH = HEIGHT = 100
MAX_SPEED = 5
ITERATIONS = 12 # Number of iterations in the game

# Helper functions
def dist(p1, p2):
    """Computes the distance between two points.
    Parameters
    ----------
    p1 : List
        Coordinates of point #1 as [x, y]
    p2 : List
        Coordinates of point #2 as [x, y]
    Returns
    -------
    float
        The distance between p1 and p2
    """
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def best_loc(positions):
    """Gets the ideal location to move to given the set of positions.
    Parameters
    ----------
    positions : List
        List of three positions [prey, agent, predator]
    Returns
    -------
    List
        The list of coordinates of the best location to move to 
        in the form [prey, agent, prey]
    """
    # Let the position of the agent be the center of the circle of radius equal to max_speed
    center = positions[1]
    radius = MAX_SPEED

    # All possible angles along the circumference
    angles = [x for x in range(361)]
    best_reward = -math.inf
    best_loc = None

    for angle in angles:
        # Get the x and y coordinates if the agent were to move at this angle
        x = radius * np.cos(np.radians(angle)) + center[0]
        if x < 0:
            x = 0
        elif x > WIDTH:
            x = WIDTH

        y = radius * np.sin(np.radians(angle)) + center[1]
        if y < 0:
            y = 0
        elif y > HEIGHT:
            y = HEIGHT

        loc = [int(x), int(y)]

        # Reward = distance to predator - distance to prey
        reward = dist(loc, positions[2]) - dist(loc, positions[0])
        # If this locations has the best reward seen yet, update
        if reward > best_reward:
            best_reward = reward
            best_loc = loc

    return best_loc

def main():
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Ideal Implementation")

    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)
    

    # Run model for n iterations
    for _ in range(ITERATIONS):
        # Prey avoids agent
        prey.avoid(agent.loc, MAX_SPEED-3)
        # Predator pursues agent
        predator.pursue(agent.loc, MAX_SPEED-3)

        # Moves the agent according to the boltzmann
        move_loc = best_loc([prey.loc, agent.loc, predator.loc])
        agent.force_move(move_loc)


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
