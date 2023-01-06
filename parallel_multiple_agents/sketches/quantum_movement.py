""" Predator-Prey Task for Movement (Quantum Approach)
This implements the Predator-Prey task for direction of movement within quantum
computing using a quantum restricted boltzmann machine sampled with quantum annealing.
"""

from data import movement_data
from rbms import movement_rbm
from samplers import quantum_sampler

TRAIN_SIZE = 100
TEST_SIZE = 30
MAX_SPEED = 5
WIDTH = HEIGHT = 100
EPOCHS = 20
LEARNING_RATE = 0.1

def main():
    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane
    dataset = movement_data.MovementData(TRAIN_SIZE, TEST_SIZE, MAX_SPEED, WIDTH, HEIGHT)

    # Initialize the QRBM sampler 
    qrbm_sampler = quantum_sampler.SamplerQRBM()

    # Define visible and hidden dimensions of the QRBM
    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15

    # Initialize the QRBM given the sampler and the dimensions
    qrbm = movement_rbm.MovementRBM(qrbm_sampler, visible_dim, hidden_dim)

    # Train the QRBM for a given number of epochs with a given learning rate
    qrbm.train(dataset.test_data_bin, EPOCHS, LEARNING_RATE)

    # Test the QRBM
    qrbm.test(dataset.test_data_bin, dataset.test_data_bin_answers)

    # Run an example
    qrbm.run_example(dataset.test_data_bin[0], dataset.test_data_bin_answers[0])

if __name__ == "__main__":
    main()
    