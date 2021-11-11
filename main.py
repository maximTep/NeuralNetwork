from PixelDrawer import PixelDrawer
from SimpleNN import SimpleNN
from FullNN import FullNN
from Funcs import *


def porog(x: float):
    return int(x>=0.5)


if __name__ == '__main__':
    drawer = PixelDrawer()

    # NN = FullNN([784, 30, 10])
    # NN.print_train = True
    # NN.weights[1] /= 100
    # NN.act_funcs = [lambda x: x, relu, sigmoid]
    # NN.der_act_funcs = [lambda x: 1, relu_der, sigmoid_der]
    # NN.run_training(400, 0.001)
    # NN.save_settings()



    inputs = [[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]
    answers = [0,
               1,
               1,
               0]
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    answers = np.array(answers)

    NN = FullNN([2, 2, 1])
    NN.print_train = True
    NN.act_funcs = [lambda x: x, lambda x: x, lambda x: x]
    NN.der_act_funcs = [lambda x: 1, lambda x: x, lambda x: 1]
    NN.run_training(inputs, answers, 50, 1)




    while True:
        img = drawer.request_image()
        img_arr = shrink_img_array(matrix_to_array(img))
        res = NN.get_result(img_arr)
        print(res)



