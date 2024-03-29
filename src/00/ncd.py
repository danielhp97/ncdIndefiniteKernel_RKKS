from ast import Raise
from tqdm import tqdm
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
    compression_type = params['compression_type']
    concatenation_type = params['concatenation']['concatenation_type']

class Ncd():
    def __init__(self) -> None:
        pass

    def _compress_image(self, i): # read array and get compressed image from position given
        """
            Takes an Image position and retuns the compresssed Image from the compressed Image npy array.
            i: image position (int)
            returns: np.asarray(PIL.Image)
        """
        array_obj = np.load("data/dataset1/kernel/compressedImagesSize.npy", allow_pickle=True) # need to check the path
        size = array_obj[i,:,:,:] # get positon
        # return number of bytes
        return size
    
    def _image_to_byte_array(self, image: Image, compression) -> bytes:
            """
                Auxiliary function to get a byte size of a PIL.Image object:
                image: PIL.Image
                returns: bytes object
            """
            imgByteArr = BytesIO()
            image.save(imgByteArr, format=compression)
            imgByteArr = imgByteArr.getvalue()
            return len(imgByteArr)

    def _size(self, img, compression):
        """
            Calculates size of Image using BytesIO()
        """
        byte_img = BytesIO()
        img.save(byte_img, format=compression)
        len = byte_img.tell()
        return len

    def _interleave_img(self, img1,img2, mode, compression):
        """
            Takes 2 Image Classes, returns a interleaved Image
        """
        r1, g1, b1 = img1.split()
        r2, g2, b2 = img2.split()
        merged_image_1 = Image.merge('{}'.format(mode), (r1,g2,b2))
        merged_image_2 = Image.merge('{}'.format(mode), (r2,g1,b2))
        merged_image_3 = Image.merge('{}'.format(mode), (r2,g2,b1))
        merged_image_4 = Image.merge('{}'.format(mode), (r2,g1,b1))
        merged_image_5 = Image.merge('{}'.format(mode), (r1,g2,b1))
        merged_image_6 = Image.merge('{}'.format(mode), (r1,g1,b2))
        listImg = [self._size(merged_image_1,compression), self._size(merged_image_2,compression), self._size(merged_image_3,compression), self._size(merged_image_4,compression), self._size(merged_image_5,compression), self._size(merged_image_6,compression)]
        return np.max(listImg)
    
    def _ncd_formula(self, len1, len2, lenconcat):
        """
            NCD specific formula. Takes only integer values
            Returns Integer with NCD value
        """
        return  (lenconcat - min(len1, len2)) / max(len1, len2)
    
    def _ncd_general(self, img1, img2):
        """
            Applies the Concatenation and NCD method mased on the compression type:
            Input: two image objects/ ND_arrays
            Returns: NCD value
        """
        # add support to paq9
        if type(img1) is Image.Image:
            pass
        else:
            img1 = Image.fromarray(img1)
        if type(img2) is Image.Image:
            pass
        else:
            img2 = Image.fromarray(img2)
        match concatenation_type:
            case "interleaved":
                concat1 = img1
                concat2 = img2
                len_concat_image = self._interleave_img(concat1,concat2, 'RGB', compression_type)
            case "NCD_shuffle":
                print("Not Implemented Yet")
                raise NotImplementedError
            case "np_concat":
                print("Currently on works")
                concat = np.asarray(concat)
                concat = np.concatenate((img1,img2), axis = params['concatenation']['concatenation_axis'])
                concat = Image.fromarray(concat)
                bytesObject = BytesIO()
                concat.save(bytesObject, params['compression_type'])
                len_concat_image = bytesObject.tell()
        ncd_value = self.ncd_calc(img1,img2, len_concat_image, params['compression_type'])
        return ncd_value

    def get_training_matrix(whole_kernel, indices, i):
        """
            Subsets kernel Matrix using the training indices. Creates a Training Kernel Matrix.
            Returns: Numpy list [indices, :][:, indices]
        """
        matrix = np.load("data/dataset{0}/kernel/kernelmatrix.npy".format(i))
        return matrix[indices, :][:, indices]

    def get_testing_matrix(whole_kernel, testing_indices, training_indices, i):
        """
            Subsets kernel Matrix using the training and test indices. Creates a Testing Kernel Matrix.
            Returns: Numpy list [training, :][:, testing]
        """
        matrix = np.load("data/dataset{0}/kernel/kernelmatrix.npy".format(i))
        return matrix[training_indices, :][:, testing_indices]

    def ncd_matrix(self, list_images):
        """
            Calculates the NCD Matrix. The function takes a list of images, takes the compressed counterpart and calculates the ncd between each element.
            Returns: Numpy List [img_array, :][:, img_array]
        """
        n = len(list_images)
        K = np.zeros((n,n))
        for i in tqdm(range(n)):
            for j in tqdm(range(i,n), leave=False):
                K[i,j] = self._ncd_general(Image.fromarray(list_images[i]),Image.fromarray(list_images[j]))
                K[j,i] = K[i,j]
        return K

    def ncd_calc(self, img1,img2,len_concat_image, compression ):
        """
            Calculates the NCD given two Image objects, the length of the concat images and the compression method for the singular images
            Returns: Float value returning the ncd
        """
        # add support to paq9
        len1 = self._image_to_byte_array(img1, compression)
        len2 = self._image_to_byte_array(img2, compression)
        return self._ncd_formula(len1,len2,len_concat_image)
