from PixelDrawer import PixelDrawer
from SimpleNN import SimpleNN
from FullNN import FullNN
from Funcs import *




if __name__ == '__main__':
    drawer = PixelDrawer()

    # NN = SimpleNN()
    # NN.print_train = True
    # NN.run_training(400, 1)

    NN = FullNN([784, 10])
    NN.print_train = True
    NN._test_train(200)


    while True:
        img = drawer.request_image()
        img_arr = shrink_img_array(matrix_to_array(img))
        res = NN.get_result(img_arr)
        print(res)



