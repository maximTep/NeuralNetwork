import numpy as np
from mnist import MNIST
from NeuralNetwork import NeuralNetwork
from Funcs import *



class FullNN(NeuralNetwork):
    def __init__(self, lay_sizes: list[int]):
        super(FullNN, self).__init__()

        for lay_size in lay_sizes:
            self.add_layer(np.ndarray(shape=[lay_size]))
        self.rand_all_weights()
        self.set_all_biases(0)
        self.print_train = False
        self.act_funcs = [self.sigmoid for _ in range(len(self.layers))]
        self.der_act_funcs = [self.sigmoid_der for _ in range(len(self.layers))]

    def get_result(self, inp: np.ndarray):
        self.set_input(inp)
        self.calculate_all_layers()
        res_layer = self.get_result_layer()
        maxi = 0
        for ind in range(len(res_layer)):
            if res_layer[ind] > res_layer[maxi]:
                maxi = ind
        return maxi


    def train(self, inp: np.ndarray, right_ans: int, alpha=1):
        res = self.get_result(inp)
        errors = self.get_errors(right_ans)
        right_res = [i == right_ans for i in range(len(self.layers[-1]))]
        for i in range(len(self.weights[1])):
            for j in range(len(self.weights[1][i])):
                delta = errors[i] * inp[j]
                self.weights[1][i][j] -= delta * alpha
        if self.print_train:
            error = self.get_sqr_error(right_ans)
            print(f'Prediction: {res}, right ans: {right_ans}, error: {error}')


    def run_training(self, iterations: int, alpha=1):
        mndata = MNIST('MNIST')
        mndata.gz = True
        images, labels = mndata.load_training()
        for i in range(iterations):
            if self.print_train: print(i, end='. ')
            self.train(shrink_img_array(images[i]), labels[i], alpha)


    def _test_train(self, iterations: int):
        images, labels = load_mnist_train()

        for it in range(iterations):
            shift = 100
            it += shift
            inp = shrink_img_array(images[it])
            right_ans = labels[it]
            res = self.get_result(inp)
            errors = self.get_errors(right_ans)
            right_res = [i == right_ans for i in range(len(self.layers[-1]))]
            if self.print_train: print(it, end='. ')

            self.back_prop(1, errors, 1)

            if self.print_train:
                error = self.get_sqr_error(right_ans)
                print(f'Prediction: {res}, right ans: {right_ans}, error: {error}')





