""" Predator-Prey Task for Attention Allocation (Quantum Approach)
This implements the Predator-Prey task for attention allocation within quantum
computing using a quantum restricted boltzmann machine sampled with quantum annealing.
"""

from data import attention_data
from qrbms import sampler, attention_qrbm

TRAIN_SIZE = 100
TEST_SIZE = 30
WIDTH = HEIGHT = 100
EPOCHS = 20
LEARNING_RATE = 0.1

def main():
    # Generate a data for training and testing given the width and the height of the 
    # coordinate plane
    dataset = attention_data.AttentionData(TRAIN_SIZE, TEST_SIZE, WIDTH, HEIGHT)

    # Initialize the QRBM sampler 
    qrbm_sampler = sampler.SamplerQRBM()

    # Define visible and hidden dimensions of the QRBM
    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15

    # Initialize the QRBM given the sampler and the dimensions
    qrbm = attention_qrbm.AttentionQRBM(qrbm_sampler, visible_dim, hidden_dim)

    # Train the QRBM for a given number of epochs with a given learning rate
    qrbm.train(dataset.test_data_bin, EPOCHS, LEARNING_RATE)

    # Test the QRBM
    qrbm.test(dataset.test_data_bin, dataset.test_data_bin_answers)

    # Run an example
    qrbm.run_example(dataset.test_data_bin[0], dataset.test_data_bin_answers[0])

if __name__ == "__main__":
    main()