import sys
import numpy as np
import pandas as pd
import yaml
import time
import paq as pq
from PIL import Image
from io import BytesIO
try:
    import lzma
except ImportError:
    from backports import lzma

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    ncd_type = params['ncd_baseline']['ncd_type']

class Ncd():
    def __init__(self) -> None:
        pass

    def compress_image(self, i): # read array and get compressed image from position given
        # read path and array
        array_obj = np.load("data/dataset1/kernel/compressedImagesSize.npy", allow_pickle=True) # need to check the path
        size = array_obj[i,:,:,:] # get positon
        # return number of bytes
        return size
    
    def image_to_byte_array(self, image: Image) -> bytes:
            imgByteArr = BytesIO()
            image.save(imgByteArr, format=image.format)
            imgByteArr = imgByteArr.getvalue()
            return imgByteArr

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
        print("O tamaho da 'lista da imagesns' que nos tá a dar é: {}    ".format(n))
        K = np.zeros((n,n))

        # implement iteration for mirror matrix
        for i in range(n): # iterate over each element
            for j in range(i,n): # for each row, iterate over each element +1
                print("linha {}; coluna {}".format(i,j))
                start_time = time.time()
                x_comp = self.compress_image(i=i) # get compressed image from position
                y_comp = self.compress_image(i=j) # get compressed image from position
                if ncd_type == "ppm_pq9":
                    x_comp = pq.compress(self.image_to_byte_array(x_comp))
                    y_comp = pq.compress(self.image_to_byte_array(y_comp))
                    len_x_comp = len(x_comp)
                    len_y_comp = len(y_comp)
                    K[i,j] = self.ncd_paq9(list_of_images[i], list_of_images[j], len_x_comp, len_y_comp) # calculate ncd
                    print("Exec time: {}".format(time.time()-start_time))
                    K[j,i] = K[i,j] # mirror result to the mirror position

                elif ncd_type == "jpeg_compression" or ncd_type == "jpeg_compressionVertical": 
                    x_comp = np.asarray(x_comp)
                    x_comp = Image.fromarray(x_comp)
                    y_comp = np.asarray(y_comp)
                    y_comp = Image.fromarray(y_comp)
                    bytex = BytesIO()
                    bytey = BytesIO()
                    x_comp.save(bytex, 'jpeg')
                    y_comp.save(bytey, 'jpeg')
                    len_x_comp = bytex.tell()
                    len_y_comp = bytey.tell()
                    if ncd_type == 'jpeg_compression':
                        K[i,j] = self.ncd_jpeg2000(list_of_images[i], list_of_images[j], len_x_comp, len_y_comp) # calculate ncd
                        print("Exec time: {}".format(time.time()-start_time))
                        K[j,i] = K[i,j] # mirror result to the mirror position
                    elif ncd_type == 'jpeg_compressionVertical':
                        K[i,j] = self.ncd_jpeg2000vertical(list_of_images[i], list_of_images[j], len_x_comp, len_y_comp) # calculate ncd
                        print("Exec time: {}".format(time.time()-start_time))
                        K[j,i] = K[i,j] # mirror result to the mirror position
        return K # return matrix

    def ncd_jpeg2000vertical(self,x,y,len_x_comp, len_y_comp):
        x_y = np.concatenate((x,y), axis = 0)
        x_y = Image.fromarray(x_y)
        bytes = BytesIO()
        x_y.save(bytes,'jpeg')
        lenx_y = bytes.tell()
        #lenx_y = sys.getsizeof(x_y)
        ncd = (lenx_y - min(len_x_comp, len_y_comp)) / \
        max(len_x_comp, len_y_comp)
        return ncd


    def ncd_jpeg2000(self,x,y,len_x_comp, len_y_comp):
        x_y = np.concatenate((x,y), axis = 0)
        x_y = Image.fromarray(x_y)
        bytes = BytesIO()
        x_y.save(bytes,'jpeg')
        lenx_y = bytes.tell()
        # compress x_y
        ncd = (lenx_y - min(len_x_comp, len_y_comp)) / \
        max(len_x_comp, len_y_comp)
        return ncd

    def ncd_paq9(self,x,y,len_x_comp, len_y_comp):
        x_y = Image.new("RGB", (600, 240), "white")
        x = Image.fromarray(x)
        y = Image.fromarray(y)
        x_y.paste(x, (0, 0))
        x_y.paste(y, (300,0))
        lenx_y = len(pq.compress(self.image_to_byte_array(x_y)))
        # compress x_y
        #lenx_y = sys.getsizeof(x_y)
        print("Size of Concatenated Image: {}".format(lenx_y))
        print("Size of x Image: {}".format(len_x_comp))
        print("Size of y Image: {}".format(len_y_comp))
        ncd = (lenx_y - min(len_x_comp, len_y_comp)) / \
        max(len_x_comp, len_y_comp)
        return ncd

