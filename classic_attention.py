from data import attention_data
import data
from rbms import sampler, attention_rbm

TRAIN_SIZE = 100
TEST_SIZE = 30
WIDTH = HEIGHT = 100
EPOCHS = 20

def main():
    dataset = attention_data.AttentionData(TRAIN_SIZE, TEST_SIZE, WIDTH, HEIGHT)

    rbm_sampler = sampler.SamplerRBM()

    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15
    rbm = attention_rbm.AttentionRBM(rbm_sampler, visible_dim, hidden_dim)

    rbm.train(dataset.test_data_bin, EPOCHS)

    rbm.test(dataset.test_data_bin, dataset.test_data_bin_answers)

if __name__ == "__main__":
    main()