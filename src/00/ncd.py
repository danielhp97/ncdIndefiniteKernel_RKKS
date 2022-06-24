import os
import numpy as np
import pandas as pd
import yaml
import time
from PIL import Image
try:
    import lzma
except ImportError:
    from backports import lzma

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']

class Ncd():
    def __init__(self) -> None:
        pass

    def compress_image(self, i): # read array and get compressed image from position given
        # read path and array
        array_obj = np.load("data/dataset1/kernel/compressedImages.npy") # need to check the path
        image = array_obj[i] # get positon
        # return image len
        return len(image)

    def ncd(self,x,y, len_x_comp, len_y_comp):
        if type(x)=="numpy.ndarray":
            pass
        else:
            #x = np.array(x)
            #y = np.array(y)
            x_y = x + y # x and y raw # 
        x_y_comp = lzma.compress(x_y)  # need to be compressed b4
        # temos que mudar oncd; só pode ser a unica com o ncd
        ncd = (len(x_y_comp) - min(len_x_comp, len_y_comp)) / \
        max(len_x_comp, len_y_comp)
        return ncd

    def get_training_matrix(whole_kernel, indices, i):
        matrix = np.load("data/dataset{0}/kernel/kernelmatrix.npy".format(i))
        return matrix[indices, :][:, indices]

    def get_testing_matrix(whole_kernel, testing_indices, training_indices, i):
        matrix = np.load("data/dataset{0}/kernel/kernelmatrix.npy".format(i))
        return matrix[training_indices, :][:, testing_indices]


    def ncd_matrix(self, list_of_images):
        # function to be called
        # objective: create function that is feeded the self ncd funtion
        # returns: ncd computed matrix
        n = len(list_of_images)
        K = np.zeros((n,n))
        # implement iteration for mirror matrix
        for i in range(n): # iterate over each element
            for j in range(i,n): # for each row, iterate over each element +1
                print("linha {}; coluna {}".format(i,j))
                start_time = time.time()
                len_x_comp = self.compress_image(i=i) # get compressed image from position
                len_y_comp = self.compress_image(i=j) # get compressed image from position
                K[i,j] = self.ncd_jpeg2000(list_of_images[i], list_of_images[j], len_x_comp, len_y_comp) # calculate ncd
                print("Exec time: {}".format(time.time()-start_time))
                K[j,i] = K[i,j] # mirror result to the mirror position
        return K # return matrix

    def ncd_vertical(self,x,y):
        x_y = np.concatenate((x,y),axis=1) # 
        x_comp = lzma.compress(x)  # compress file 1
        y_comp = lzma.compress(y)  # compress file 2
        x_y_comp = lzma.compress(x_y)  # compress file concatenated
        ncd = (len(x_y_comp) - min(len(x_comp), len(y_comp))) / \
        max(len(x_comp), len(y_comp))
        return ncd


    def ncd_jpeg2000(self,x,y,len_x_comp, len_y_comp):
        x_y = np.concatenate((x,y), axis = 0)
        ncd = (len(x_y) - min(len_x_comp, len_y_comp)) / \
        max(len_x_comp, len_y_comp)
        return ncd

