from data import attention_data
from qrbms import sampler, attention_qrbm


TRAIN_SIZE = 100
TEST_SIZE = 30
WIDTH = HEIGHT = 100
EPOCHS = 20

def main():
    dataset = attention_data.AttentionData(TRAIN_SIZE, TEST_SIZE, WIDTH, HEIGHT)

    qrbm_sampler = sampler.SamplerQRBM()

    visible_dim = len(dataset.train_data_bin[0])
    hidden_dim = 15
    qrbm = attention_qrbm.AttentionQRBM(qrbm_sampler, visible_dim, hidden_dim)

    qrbm.train(dataset.test_data_bin, EPOCHS, 0.1)

    qrbm.test(dataset.test_data_bin, dataset.test_data_bin_answers)

if __name__ == "__main__":
    main()