import numpy as np
import random
import math
from Funcs import *

np.random.seed(random.randint(1, 300))


class NeuralNetwork:
    def __init__(self):
        self.layers:list[np.ndarray] = []
        self.weights:list[np.ndarray] = []
        self.biases = []
        self.act_funcs = []
        self.der_act_funcs = []
        self.print_train = False

    def size_of_layers(self):
        return len(self.layers)

    def set_input(self, inp: np.ndarray):
        inp = np.array(inp)
        self.layers[0] = inp

    def add_layer(self, layer: np.ndarray):
        self.layers.append(layer)
        self.weights.append([])
        self.biases.append([])

    def rand_weights(self, lay_num):
        if lay_num == 0:
            raise ValueError('No previous layer')
        weights = np.random.rand(len(self.layers[lay_num]), len(self.layers[lay_num - 1])) - 0.5
        self.weights[lay_num] = weights

    def rand_biases(self, lay_num):
        biases = np.ndarray(shape=(len(self.layers[lay_num])))
        for i in range(len(self.layers[lay_num])):
            biases[i] = random.randint(-20, 20) / random.randint(1, 5)
        self.biases[lay_num] = biases

    def set_all_biases(self, value: float):
        for lay_num in range(1, len(self.layers)):
            biases = np.ndarray(shape=[(len(self.layers[lay_num]))])
            for i in range(len(self.layers[lay_num])):
                biases[i] = value
            self.biases[lay_num] = biases

    def rand_all_weights(self):
        np.random.seed(1)
        for i in range(1, len(self.layers)):
            self.rand_weights(i)

    def rand_all_biases(self):
        for i in range(1, len(self.layers)):
            self.rand_biases(i)

    def calculate_layer(self, lay_num):
        if lay_num == 0:
            raise ValueError('No previous layer')
        layer = np.dot(self.weights[lay_num], self.layers[lay_num - 1]) + self.biases[lay_num]
        for i in range(len(layer)):
            layer[i] = self.act_funcs[lay_num](layer[i])
        self.layers[lay_num] = layer

    def calculate_all_layers(self):
        for i in range(1, len(self.layers)):
            self.calculate_layer(i)

    def get_result_layer(self):
        return self.layers[-1]

    def save_settings(self):
        f = open('NeuralSettings.txt', 'w')
        f.write(str(len(self.weights)) + '\n')

        for i in range(1, len(self.weights)):
            f.write(str(len(self.weights[i])) + ' ' + str(len(self.weights[i][0])) + '\n')
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    f.write(str(self.weights[i][j][k]) + ' ')
                f.write('\n')
            f.write('\n')

        for i in range(1, len(self.biases)):
            f.write(str(len(self.biases[i])) + '\n')
            for j in range(len(self.biases[i])):
                f.write(str(self.biases[i][j]) + ' ')
            f.write('\n')


        f.close()

    def set_settings(self, file_name: str):
        f = open(file_name, 'r')
        num_of_layers = 0
        lines = f.read().split('\n')
        i = 0
        while i < len(lines):
            if lines[i] == ' ':
                lines.pop(i)
            else:
                i += 1
        it = 0
        num_of_layers = int(lines[0])
        it += 1
        for i in range(1, num_of_layers):
            n, m = map(int, lines[it].split(' '))
            new_weight = np.ndarray(shape=(n, m))
            it += 1
            for j in range(n):
                nums_in_row = lines[it].split(' ')
                nums_in_row.pop()
                row = list(map(float, nums_in_row))
                it += 1
                for k in range(m):
                    new_weight[j][k] = row[k]
            self.weights[i] = new_weight
            it += 1
        for i in range(1, num_of_layers):
            n = int(lines[it])
            new_bias = np.ndarray(shape=n)
            it += 1
            nums_in_row = lines[it].split(' ')
            nums_in_row.pop()
            row = list(map(float, nums_in_row))
            it += 1
            for j in range(n):
                new_bias[j] = row[j]
            self.biases[i] = new_bias




        f.close()

    def get_errors(self, ans: int):
        right_res = [int(i == ans) for i in range(len(self.layers[-1]))]
        return [(res - right_res[ind]) for ind, res in enumerate(self.get_result_layer())]

    def get_sqr_error(self, ans: int):
        errors = self.get_errors(ans)
        summ = 0
        for err in errors:
            summ += err**2
        return summ

    def get_result(self, inp: np.ndarray):
        self.set_input(inp)
        self.calculate_all_layers()
        res_layer = self.get_result_layer()
        maxi = 0
        for ind in range(len(res_layer)):
            if res_layer[ind] > res_layer[maxi]:
                maxi = ind
        return maxi

    def back_prop(self, lay_num: int, err_lst: list[float], alpha: float):
        if lay_num == 0: raise ValueError('No prev layer')
        lay = self.layers[lay_num]
        prev_lay = self.layers[lay_num-1]
        back_errors = []
        for j in range(len(prev_lay)):
            deltas = 0
            for i in range(len(lay)):
                delta = err_lst[i]
                scaled_delta = delta * prev_lay[j] * self.der_act_funcs[lay_num](lay[i])
                deltas += scaled_delta
                self.weights[lay_num][i][j] -= scaled_delta * alpha
            back_errors.append(deltas)
        back_errors = np.array(back_errors)
        return back_errors

    def run_training_mnist(self, iterations: int, alpha=1):
        images, labels = load_mnist_train()
        for it in range(iterations):
            shift = 0
            it += shift
            inp = shrink_img_array(images[it])
            right_ans = labels[it]
            res = self.get_result(inp)
            AAAAA__res_lay = self.get_result_layer()
            errors = np.array(self.get_errors(right_ans))
            right_res = [i == right_ans for i in range(len(self.layers[-1]))]
            if self.print_train: print(it, end='. ')
            errors *= self.der_act_funcs[-1](self.layers[-1])
            delta = errors
            for lay_num in reversed(range(1, len(self.layers))):
                # errors = self.back_prop(lay_num, errors, 1)
                self.weights[lay_num] -= (alpha * self.layers[lay_num-1][:, np.newaxis].dot(delta[:, np.newaxis].T)).T
                delta = delta.dot(self.weights[lay_num]) * self.der_act_funcs[lay_num-1](self.layers[lay_num-1])

            if it> 1000:
                errors = np.array(self.get_errors(right_ans))
                lyy = self.layers[-1]
                aaaa = 10

            if self.print_train:
                error = self.get_sqr_error(right_ans)
                print(f'Prediction: {res}, right ans: {right_ans}, error: {error}')




    def run_training(self, inputs: list[np.ndarray], answers: np.ndarray, iterations: int, alpha=1):
        for it in range(iterations):
            shift = 0
            it += shift
            inp = inputs[it%len(inputs)]
            right_ans = answers[it%len(answers)]
            res = self.get_result(inp)
            res_layer = self.layers[-1]
            errors = res_layer - right_ans
            if self.print_train: print(it, end='. ')
            errors *= self.der_act_funcs[-1](self.layers[-1])
            delta = errors
            for lay_num in reversed(range(1, len(self.layers))):
                # errors = self.back_prop(lay_num, errors, 1)
                self.weights[lay_num] -= (alpha * self.layers[lay_num-1][:, np.newaxis].dot(delta[:, np.newaxis].T)).T
                delta = delta.dot(self.weights[lay_num]) * self.der_act_funcs[lay_num-1](self.layers[lay_num-1])

            if self.print_train:
                error = errors**2
                print(f'Prediction: {res}, right ans: {right_ans}, error: {error}')








