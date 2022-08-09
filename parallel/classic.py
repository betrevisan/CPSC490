""" Complete Predator-Prey Task (Classical Approach)
This implements the complete Predator-Prey task within classical computing using a
restricted boltzmann machine.
"""

import time
from data import data
from rbms import rbm
from samplers import classic_sampler
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod
from metrics import metrics as metrics_mod

TRAIN_SIZE = 1000
TEST_SIZE = 100
MAX_SPEED = 5
WIDTH = HEIGHT = 100
EPOCHS = 25
LEARNING_RATE = 0.1
ITERATIONS = 12 # Number of iterations in the game

def main():
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Parallel Classical Implementation")

    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane and the maximum speed
    dataset = data.Data(TRAIN_SIZE, TEST_SIZE, MAX_SPEED, WIDTH, HEIGHT)

    # Initialize the RBM sampler, which will be used in the RBM
    rbm_sampler = classic_sampler.SamplerRBM()

    # Define visible and hidden dimensions of the RBM
    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15

    # Initialize the RBM given the sampler and the dimensions
    rbm_model = rbm.RBM(rbm_sampler, visible_dim, hidden_dim)

    # Start the training timer (must be adjusted from seconds to microseconds)
    training_start_time = time.time() * 1000000
    # Train the RBM for a given number of epochs with a given learning rate
    rbm_model.train(dataset.test_data_bin, EPOCHS, 0.1)
    # Record time elapsed during training
    metrics.training_time += (time.time() * 1000000 - training_start_time)

    # Test the RBM
    rbm_model.test(dataset.test_data_bin, dataset.test_data_bin_answers)

    # Run an example
    rbm_model.run_example(dataset.test_data_bin[0], dataset.test_data_bin_answers[0])

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

        # Start the movement timer
        decision_start_time = time.time() * 1000000
        # Moves the agent according to the boltzmann
        rbm_model.move_locs(agent, prey.loc, predator.loc, MAX_SPEED)
        # Record time elapsed during decision
        metrics.decision_time += (time.time() * 1000000 - decision_start_time)

    # Add general metrics
    metrics.w = WIDTH
    metrics.h = HEIGHT
    metrics.iterations = ITERATIONS
    metrics.visible_dim = visible_dim
    metrics.hidden_dim = hidden_dim
    metrics.training_size = TRAIN_SIZE
    metrics.test_size = TEST_SIZE
    metrics.max_speed = MAX_SPEED
    metrics.learning_rate = LEARNING_RATE
    metrics.epochs = EPOCHS

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
