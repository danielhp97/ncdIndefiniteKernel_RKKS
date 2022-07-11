from pyexpat import model
from re import L
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import numpy as np
import pandas as pd
import pickle
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

def image_filtering(img, normal=False):
    #filtered = img.filter(MedianFilter(size=9))
    resized = img.resize((300,240))
    if normal:
        resized = resized.filter(BLUR)
        resized = resized.filter(MedianFilter)
    return resized
     
def ncd_preparation(dataset, ncd_type):
    data = dataset
    np_images = []
    data['Path'] = 'data' + data['Path'].astype(str)
    path_list = data['Path']
    counter_x = 0
    for i in path_list:
        if ncd_type == 'jpeg_compressionVertical':
            start_time = time.time()
            print("imagem x: {0}".format(counter_x))
            temp = Image.open(i)
            temp = image_filtering(temp)
            # write image as jpeg to temp (with quality parameters)
            temp = temp.save("temp/ncd_tempImg.jpg")
            # load image as temp
            temp = Image.open("temp/ncd_tempImg.jpg")
            temp = np.array(temp)
            # transform to 16 unsigned array
            np_images.append(temp)
            counter_x +=1
            print("Tempo de execucao: {}".format(time.time()-start_time))
        elif ncd_type == 'jpeg_compression':
            start_time = time.time()
            print("imagem x: {0}".format(counter_x))
            temp = Image.open(i)
            temp = image_filtering(temp)
            # write image as jpeg to temp (with quality parameters)
            temp = temp.save("temp/ncd_tempImg.jpg")
            # load image as temp
            temp = Image.open("temp/ncd_tempImg.jpg")
            temp = np.array(temp)
            # transform to 16 unsigned array
            np_images.append(temp)
            counter_x +=1
            print("Tempo de execucao: {}".format(time.time()-start_time))
        else:
            start_time = time.time()
            print("imagem x: {0}".format(counter_x))
            temp = Image.open(i)
            temp = image_filtering(temp)
            temp = np.asanyarray(temp, dtype=object)
            temp = lzma.compress(temp)
            np_images.append(temp)
            counter_x +=1
            print("Tempo de execucao: {}".format(time.time()-start_time))
    return np_images


def ncd_kernel(x_images):
    n1 = len(x_images)
    n2 = len(x_images)
    k = np.zeros((n1,n2))
    calc = Ncd()
    k = calc.ncd_matrix(x_images)
    return k




def dir_cleaning(x):
    #deleting
    if os.path.exists("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x)):
        for f in os.path.os.listdir("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x)):
            # ver se e directoria
            if os.path.isdir(os.path.join("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x),f)):
                for f2 in os.path.os.listdir("data/02/dataset" + str(dataset_number) + "/train/{0}/{1}".format(x,f)):
                    os.remove(os.path.join("data/02/dataset" + str(dataset_number) + "/train/{0}/{1}".format(x,f), f2))
                os.rmdir(os.path.join("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x),f))
            elif os.path.isfile(os.path.join("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x),f)):
                os.remove(os.path.join("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x),f))
        os.rmdir("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x))
    #### writing
    os.mkdir("data/02/dataset" + str(dataset_number) + "/train/{0}".format(x))

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
    print("Kernel Matrix:")
    print(pd.DataFrame(kernel_matrix))
    original_img1 = com_images[2]
    original_img2 = com_images[34]
    calc = Ncd()
    print("________ compressed images test_________")
    #length image 1
    img1 = com_images[2]
    img1 = np.asarray(img1)
    img1 = Image.fromarray(img1)
    byteimg1 = BytesIO()
    img1.save(byteimg1, 'jpeg')
    lenimg1 = byteimg1.tell()


    #length image 2
    img2 = com_images[34]
    img2 = np.asarray(img2)
    img2 = Image.fromarray(img2)
    byteimg2 = BytesIO()
    img2.save(byteimg2, 'jpeg')
    lenimg2 = byteimg2.tell()


    print("Length of Image 1: {}".format(lenimg1))
    print("Length of Image 2: {}".format(lenimg2))

    print("________ ncd test_________")
    test = calc.ncd_jpeg2000(img1,img2,lenimg1,lenimg2)
    print("Ncd with Horizontal Concatenation: {}".format(test))
    test = calc.ncd_jpeg2000vertical(img1,img2,lenimg1,lenimg2)
    print("Ncd with Vertical Concatenation: {}".format(test))
