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
TEST_SIZE = 30
MAX_SPEED = 5
WIDTH = HEIGHT = 100
EPOCHS = 25
LEARNING_RATE = 0.1
ITERATIONS = 12 # Number of iterations in the game
PREY_COUNT = 2
PREDATOR_COUNT = 2

def main():
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Parallel Classical Implementation")

    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane and the maximum speed
    dataset = data.Data(TRAIN_SIZE, TEST_SIZE, MAX_SPEED, WIDTH, HEIGHT, PREY_COUNT, PREDATOR_COUNT)

    # Initialize the RBM sampler, which will be used in the RBM
    rbm_sampler = classic_sampler.SamplerRBM()

    # Define visible and hidden dimensions of the RBM
    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15

    cutoff = 14*(PREY_COUNT+PREDATOR_COUNT+1)

    # Initialize the RBM given the sampler and the dimensions
    rbm_model = rbm.RBM(rbm_sampler, visible_dim, hidden_dim, cutoff, PREY_COUNT, PREDATOR_COUNT)

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
    prey1 = prey_mod.Prey(WIDTH, HEIGHT, 35, 35)
    prey2 = prey_mod.Prey(WIDTH, HEIGHT, 25, 25)
    predator1 = predator_mod.Predator(WIDTH, HEIGHT, 70, 70)
    predator2 = predator_mod.Predator(WIDTH, HEIGHT, 95, 95)

    # Run model for n iterations
    for _ in range(ITERATIONS):
        # Prey avoids agent
        prey1.avoid(agent.loc, MAX_SPEED-4)
        prey2.avoid(agent.loc, MAX_SPEED-4)
        # Predator pursues agent
        predator1.pursue(agent.loc, MAX_SPEED-4)
        predator2.pursue(agent.loc, MAX_SPEED-4)

        # Start the movement timer
        decision_start_time = time.time() * 1000000
        # Moves the agent according to the boltzmann
        rbm_model.move_locs(agent, prey1.loc, prey2.loc, predator1.loc, predator2.loc, MAX_SPEED)
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

    # Add prey to metrics
    metrics.prey1_alive = prey1.alive
    metrics.prey1_loc_trace = prey1.loc_trace

    metrics.prey2_alive = prey2.alive
    metrics.prey2_loc_trace = prey2.loc_trace

    # # Add predator to metrics
    metrics.predator1_feasted = predator1.feasted
    metrics.predator1_loc_trace = predator1.loc_trace

    metrics.predator2_feasted = predator2.feasted
    metrics.predator2_loc_trace = predator2.loc_trace

    # print(predator2.loc_trace)

    # Add attention trace to metrics
    metrics.attention_trace = agent.attn_trace

    # Display metrics
    print(metrics)

    return

if __name__ == "__main__":
    main()
