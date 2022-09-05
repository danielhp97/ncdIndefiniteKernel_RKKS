from pyexpat import model
from re import L
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import paq as pq
from io import BytesIO
import time
from PIL import Image
from PIL.ImageFilter import MedianFilter, BLUR 
from learning_algorithms import  SquareHingeKreinSVM, kernel_matrix
from ncd import  Ncd
try:
    import lzma
except ImportError:
    from backports import lzma
     
if __name__ == "__main__":
    #i = sys.argv[1]
    #i = str(i)
    # compute the kernel
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    ncd_type = params['ncd_baseline']['ncd_type']
    
    com_images = np.load("data/dataset1/kernel/compressedImagesSize.npy", allow_pickle=True)
    kernel_matrix = np.load("data/dataset1/kernel/kernelmatrix.npy")
    
    #ax = sns.heatmap(kernel_matrix)
    #sns.color_palette("rocket", as_cmap=True)
    #plt.savefig('kernelHeatmap08-1.png', bbox_inches='tight')
    #plt.show()

    original_img1 = com_images[2,:,:,:]
    original_img12 = com_images[124,:,:,:]
    original_img2 = com_images[3,:,:,:]
    calc = Ncd()
    print("____________________PNG-only Tests___________________________")
    #length image 1
    img1 = com_images[2,:,:,:]
    img1 = np.asarray(img1)
    img1 = Image.fromarray(img1)
    byteimg1 = BytesIO()
    img1.save(byteimg1, 'PNG')
    lenimg1 = byteimg1.tell()
    # image 12
    img12 = com_images[124,:,:,:]
    img12 = np.asarray(img12)
    img12 = Image.fromarray(img12)
    byteimg12 = BytesIO()
    img12.save(byteimg12, 'PNG')
    lenimg12 = byteimg12.tell()
    #length image 3
    img2 = com_images[3,:,:,:]
    img2 = np.asarray(img2)
    img2 = Image.fromarray(img2)
    byteimg2 = BytesIO()
    img2.save(byteimg2, 'PNG')
    lenimg2 = byteimg2.tell()

    img1.save("temp/img1.jpeg") #tiff: 216kb;  png: 81kb; jpeg: 12kb
    img12.save("temp/img12.jpeg") #tiff: 216kb; png: 181kb; jpeg: 10kb
    img2.save("temp/img2.jpeg") #tiff: 216kb; png: 87kb; 26kb
    print("Length of Image 1: {}".format(lenimg1))
    print("Length of Image 12: {}".format(lenimg12))
    print("Length of Image 2: {}".format(lenimg2))

    #TIFF is not a good format as it is losslesss: all the files are the same size.

    # concatenate the images and then calculate the length
    concat_img = np.concatenate((original_img1, original_img1), axis=0)
    concat_img = np.asarray(concat_img)
    concat_img = Image.fromarray(concat_img)
    byteconcat_img = BytesIO()
    concat_img.save(byteconcat_img, 'PNG')
    lenconcat_img = byteconcat_img.tell()

    ncd = (lenconcat_img - min(lenimg1, lenimg1)) / \
    max(lenimg1, lenimg1)

    print("Same image NCD test: {}".format(ncd))

    # concatenate the images and then calculate the length
    concat_img = np.concatenate((original_img1, original_img12), axis=0)
    concat_img = np.asarray(concat_img)
    concat_img = Image.fromarray(concat_img)
    byteconcat_img = BytesIO()
    concat_img.save(byteconcat_img, 'PNG')
    lenconcat_img = byteconcat_img.tell()
    ncd = (lenconcat_img - min(lenimg1, lenimg12)) / \
    max(lenimg1, lenimg12)
    print("Same class NCD test: {}".format(ncd))

    # concatenate the images and then calculate the length
    concat_img = np.concatenate((original_img1, original_img2), axis=0)
    concat_img = np.asarray(concat_img)
    concat_img = Image.fromarray(concat_img)
    byteconcat_img = BytesIO()
    concat_img.save(byteconcat_img, 'PNG')
    lenconcat_img = byteconcat_img.tell()

    ncd = (lenconcat_img - min(lenimg1, lenimg2)) / \
    max(lenimg1, lenimg2)
    print("Same Different class NCD test: {}".format(ncd)) 



    print("____________________Interleaving instead of Concatenating Images Test___________________________")


    def size(img):
        """
            Calculates size of Image using BytesIO()
        """
        byte_img = BytesIO()
        img.save(byte_img, 'PNG')
        len = byte_img.tell()
        return len


    def interleave_img(img1,img2, mode):
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
        listImg = [size(merged_image_1), size(merged_image_2), size(merged_image_3), size(merged_image_4), size(merged_image_5), size(merged_image_6)]
        return np.max(listImg)

    # median:
    #Same image NCD test: 0.0
    #Same class NCD test: 0.39079853827640265
    #Same Different class NCD test: 0.24036014763531327

    # mean:
    #Same image NCD test: 0.0
    #Same class NCD test: 0.38944979281208825
    #Same Different class NCD test: 0.23800081000741238

    # min:
    #Same image NCD test: 0.0
    #Same class NCD test: 0.3167987811609883
    #Same Different class NCD test: 0.2211893354118429

    # max:
    #Same image NCD test: 0.0
    #Same class NCD test: 0.4590463467950275
    #Same Different class NCD test: 0.24563856857935398



        
    # concatenate the images and then calculate the length
    lenconcat_img = interleave_img(img1, img1, 'RGB')
    ncd = (lenconcat_img - min(lenimg1, lenimg1)) / \
    max(lenimg1, lenimg1)

    print("Same image NCD test: {}".format(ncd))

    # concatenate the images and then calculate the length
    lenconcat_img = interleave_img(img1, img12, 'RGB')
    ncd = (lenconcat_img - min(lenimg1, lenimg12)) / \
    max(lenimg1, lenimg12)
    print("Same class NCD test: {}".format(ncd))

    # concatenate the images and then calculate the length
    lenconcat_img = interleave_img(img1, img2, 'RGB')
    ncd = (lenconcat_img - min(lenimg1, lenimg2)) / \
    max(lenimg1, lenimg2)
    print("Same Different class NCD test: {}".format(ncd)) 


    # we need to test the kernel matrix for this
    # we are testing first with the max values: the biggest difference between classes
    # incorporate this on the kernel_calc.py
