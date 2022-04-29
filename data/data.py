import numpy as np

class Data:
    np.random.seed(11)

    def __init__(self, train_size, test_size, width, height):
        self.train_size = train_size
        self.test_size = test_size
        self.width = width
        self.height = height
        # self.train_data = self.generate_train_data()
        # self.test_data = self.generate_test_data()
        # self.train_data_bin = self.prepare_data(self.train_data)
        # self.test_data_bin = self.prepare_data(self.test_data)
        # self.train_data_bin_answers = self.prepare_train_answers()
        # self.test_data_bin_answers = self.prepare_test_answers()