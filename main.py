from PixelDrawer import PixelDrawer
from SimpleNN import SimpleNN
from FullNN import FullNN
from Funcs import *




if __name__ == '__main__':
    drawer = PixelDrawer()

    NN = FullNN([784, 16, 10])
    NN.print_train = True
    NN.run_training(2000, 0.05)
    NN.save_settings()


    while True:
        img = drawer.request_image()
        img_arr = shrink_img_array(matrix_to_array(img))
        res = NN.get_result(img_arr)
        print(res)



