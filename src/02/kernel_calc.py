from tqdm import tqdm
from re import L
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import paq as pq
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
    if resized.mode not in ("RGB"):
            resized.convert('RGB')
    if normal:
        resized = resized.filter(BLUR)
        resized = resized.filter(MedianFilter)
    return resized
     
def ncd_preparation(dataset, compression):
    data = dataset
    np_images = []
    data['Path'] = 'data' + data['Path'].astype(str)
    path_list = data['Path']
    for i in tqdm(path_list):
        start_time = time.time()
        temp = Image.open(i)
        temp.save("temp/KernelPreparationImage.{}".format(compression))
        temp = Image.open("temp/KernelPreparationImage.{}".format(compression))
        temp = image_filtering(temp)
        temp = np.asarray(temp)
        np_images.append(temp)
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

def BW_check(img_list):
    """
        Takes a list of images path, checks if it is black and white, if it is, converts it to RGD it.
    """
    for p in tqdm(img_list):
        img = Image.open("data" + p)
        if img.mode not in ("RGB"):
            img.convert('RGB')
            img.save("data" + p)
        else:
            pass




if __name__ == "__main__":
    # compute the kernel
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    compression = params['compression_type']
    # what to do:
    # open /data/dataset1/labels.csv into a dataframe
    prepared_dataframe = pd.DataFrame(columns= ["ImageName","Class","Path"])
    counter = 1
    f = open("data/dataset{}/labels.csv".format(str(dataset_number)), "r")
    for line in tqdm(f):
        entry = line[2:-1]
        entry = entry[0:-2]
        entry = entry.split(",")
        df_length= len(prepared_dataframe)
        prepared_dataframe.loc[df_length] = entry
        counter += 1
    f.close()
    is_filtered = prepared_dataframe[prepared_dataframe["Class"].isin([str(" '" + params['class1'] + "'"), str(" '" + params['class2'] + "'")])]
    #cleaning dataset
    is_filtered['Path'] = is_filtered['Path'].str.lstrip(" '")
    is_filtered['Path'] = is_filtered['Path'].str.rstrip("'")
    # ncd_preparation + resize images # remove black and white images
    print("Checking for BW Images...")
    BW_check(is_filtered['Path'])
    print("Finished preparing the Label file")

    print("Started compressing dataset")
    np_images = ncd_preparation(is_filtered, compression)
    np.save('data/dataset{0}/kernel/compressedImagesSize'.format(dataset_number), np_images)
    print("Finished compressing")
    # ncd_kernel
    np_images= np.load('data/dataset{0}/kernel/compressedImagesSize.npy'.format(dataset_number), allow_pickle=True)
    #
    print("Starting kernel calculation")
    kernel = ncd_kernel(np_images)
    # output kernel
    # pre compress images
    np.save('data//dataset{0}/kernel/kernelmatrix'.format(dataset_number), kernel)
    kernel= np.load('data/dataset{0}/kernel/kernelmatrix.npy'.format(dataset_number), allow_pickle=True)
    print(pd.DataFrame(kernel))
    print("Saving KernelHeatMap...")
    heatmap = sns.heatmap(kernel)
    fig = heatmap.get_figure()
    fig.savefig("plots/kernelHeatmap.png")
    print("Done!")

