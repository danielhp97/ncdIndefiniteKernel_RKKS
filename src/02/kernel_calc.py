from pyexpat import model
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import numpy as np
import pandas as pd
import pickle
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
        if ncd_type == 'jpeg_compression':
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
    # what to do:
    # open /data/dataset1/labels.csv into a dataframe
    prepared_dataframe = pd.DataFrame(columns= ["ImageName","Class","Path"])
    counter = 1
    f = open("data/dataset{}/labels.csv".format(str(dataset_number)), "r")
    for line in f:
        entry = line[2:-1]
        entry = entry[0:-2]
        entry = entry.split(",")
        df_length= len(prepared_dataframe)
        prepared_dataframe.loc[df_length] = entry
        print("finished adding line {}".format(counter))
        counter += 1
    f.close()
    is_filtered = prepared_dataframe[prepared_dataframe["Class"].isin([str(" '" + params['class1'] + "'"), str(" '" + params['class2'] + "'")])]
    #cleaning dataset
    is_filtered['Path'] = is_filtered['Path'].str.lstrip(" '")
    is_filtered['Path'] = is_filtered['Path'].str.rstrip("'")
    # ncd_preparation + resize images # remove black and white images
    print(" finished preparing label file")
    print("started compressing dataset")
    np_images = ncd_preparation(is_filtered, ncd_type)
    np.save('data/dataset{0}/kernel/compressedImages'.format(dataset_number), np_images)
    print(len(np_images))
    print("Finished compressing")
    # ncd_kernel
    np_images= np.load('data/dataset{0}/kernel/compressedImages.npy'.format(dataset_number))


    print("Starting kernel calculation")
    kernel = ncd_kernel(np_images)
    # output kernel
    # pre compress images
    np.save('data//dataset{0}/kernel/kernelmatrix'.format(dataset_number), kernel)