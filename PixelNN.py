from NeuralNetwork import *
import pygame
import numpy as np
from Funcs import *






image = pygame.image.load('Data\\0-0.png')
grid = pixel_reader(image, 280, 280, 10)
pixels = matrix_to_array(grid)

NN = NeuralNetwork()

NN.add_layer(pixels)
NN.add_layer(np.zeros(shape=16))
NN.add_layer(np.zeros(shape=16))
NN.add_layer(np.zeros(shape=10))


# NN.rand_all_weights()
# NN.rand_all_biases()
#
# # print(NN.weights[1])
# NN.calculate_all_layers()
#
# result = np.argmax(NN.layers[-1])
# #print(result)
# # NN.save_settings()
NN.set_settings('NeuralSettings.txt')

print(NN.biases)



