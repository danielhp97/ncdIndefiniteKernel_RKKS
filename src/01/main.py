import numpy as np
import pandas as pd
import yaml
import os
import sys
import csv

from PIL import Image
from PIL.ImageFilter import MedianFilter, BLUR

def get_dir(path):
    # get csv for x and y 
    x = pd.read_csv(os.path.join(path, "x.csv"), header=None, quotechar="'", sep=',')
    y = pd.read_csv(os.path.join(path, "y.csv"), header=None, quotechar="'", sep=',')
    x_list = x.iloc[:,0].values.tolist()
    # join them
    # make list of x paths
    # return both list of paths and labels 
    return x_list, y.values.tolist() 

def rotation(img):
    rotation1 = img.rotate(2.5)
    rotation2 = rotation1.rotate(2.5)
    rotation3 = rotation2.rotate(2.5)
    return rotation1, rotation2, rotation3

def image_filtering(img, normal=False):
    #filtered = img.filter(MedianFilter(size=9))
    resized = img.resize((300,240))
    if normal:
        resized = resized.filter(BLUR)
        resized = resized.filter(MedianFilter)
    return resized

def dir_cleaning(x):
    #deleting
    if os.path.exists("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x)):
        for f in os.path.os.listdir("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x)):
            # ver se e directoria
            if os.path.isdir(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x),f)):
                print(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x),f))
                for f2 in os.path.os.listdir("data/01/dataset" + str(dataNumber) + "/train/{0}/{1}".format(x,f)):
                    print(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}/{1}".format(x,f), f2))
                    os.remove(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}/{1}".format(x,f), f2))
                os.rmdir(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x),f))
            elif os.path.isfile(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x),f)):
                os.remove(os.path.join("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x),f))
        os.rmdir("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x))
    if os.path.exists("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x)):
        for f in os.path.os.listdir("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x)):
            # ver se e directoria
            if os.path.isdir(os.path.join("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x),f)):
                for f2 in os.path.os.listdir("data/01/dataset" + str(dataNumber) + "/test/{0}/{1}".format(x,f)):
                    os.remove(os.path.join("data/01/dataset" + str(dataNumber) + "/test/{0}/{1}".format(x,f), f2))
                os.rmdir(os.path.join("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x),f))
            elif os.path.isfile(os.path.join("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x),f)):
                os.remove(os.path.join("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x),f))
        os.rmdir("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x))
    #### writing
    os.mkdir("data/01/dataset" + str(dataNumber) + "/train/{0}".format(x))
    os.mkdir("data/01/dataset" + str(dataNumber) + "/test/{0}".format(x))
    os.mkdir("data/01/dataset" + str(dataNumber) + "/train/{0}/img_dump".format(x))
    os.mkdir("data/01/dataset" + str(dataNumber) + "/test/{0}/img_dump".format(x))
    if os.path.exists("data/01/dataset" + str(dataNumber) + "/train/{}/img_dump/labels.csv"):
        os.remove("data/01/dataset" + str(dataNumber) + "/train/{}/img_dump/labels.csv")
    if os.path.exists("data/01/dataset" + str(dataNumber) + "/test/{}/img_dump/labels.csv"):
        os.remove("data/01/dataset" + str(dataNumber) + "/test/{}/img_dump/labels.csv")

# iterate over each division:
#   transformations
# export to data/01/dataset" + str(dataNumber) + "/train/i

if __name__ == "__main__":
    i = sys.argv[1]
    i = str(i)

    #save image token
    token = True
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    # select the datasets to use
    dataNumber = params['dataset']
    # load images and extract features
    test_dir = os.path.join("data","00","dataset" + str(dataNumber),"test",i)
    train_dir = os.path.join("data","00","dataset" + str(dataNumber),"train",i)
    # initialise path to store updated images
    new_train_dir = os.path.join("data","01","dataset" + str(dataNumber),"train",i, "img_dump")
    new_test_dir = os.path.join("data","01","dataset" + str(dataNumber),"test",i, "img_dump")

    #clean dir
    dir_cleaning(i)
    # 


    imagelist_train, y_train = get_dir(train_dir)
    imagelist_test , y_test = get_dir(test_dir)
    base_dir = "data"
    np_array_train = []
    np_array_test = []
    counter = -1
    f = open(new_train_dir + "/" + 'labels.csv', 'a')
    for p in imagelist_train:
        counter += 1# counter for test 
        p = p.replace("'", "")
        p = p.replace(" ", "")
        p_name = p.split("/")[-1]#get the image name
        temp = Image.open(base_dir + p)
        temp = image_filtering(temp)
        #create file goes here
        if token:
            temp.save(new_train_dir + "/" + p_name)
            #save a file com o nome da image e a label
            line = y_train[counter] # get the line corresponding to the current image
            f = open(new_train_dir + "/" + 'labels.csv', 'a')
            writer = csv.writer(f, delimiter=",")
            writer.writerow((str(p_name), line[0]))
        else:
            temp = np.asanyarray(temp, dtype=object)
            np_array_train.append(temp)
    f.close()
    counter = -1
    #create file goes here
    f = open(new_test_dir + "/" + 'labels.csv', 'a')
    for p in imagelist_test:
        counter += 1# counter for test 
        p = p.replace("'", "")
        p = p.replace(" ", "")
        p_name = p.split("/")[-1]#get the image name
        temp = Image.open(base_dir + p)
        temp = image_filtering(temp, normal=True)
        if token:
            temp.save(new_test_dir + "/" + p_name)
            #save a file com o nome da image e a label
            line = y_test[counter] # get the line corresponding to the current image
            writer = csv.writer(f)
            writer.writerow((str(p_name), line[0]))
        #temp = image_filtering(temp)
        else:
            temp = np.asanyarray(temp, dtype=object)
            np_array_test.append(temp)
    f.close()
    # save array data.
    # save as npy
    if token:
        print("Images saved to directories:")
        print("'{0}' and '{1}'".format(new_train_dir, new_test_dir))
    else:
        np.save(os.path.join("data","01","dataset" + str(dataNumber),"train",i,"array"), np_array_train)
        np.save(os.path.join("data","01","dataset" + str(dataNumber),"train",i,"y"), y_train)
        np.save(os.path.join("data","01","dataset" + str(dataNumber),"test",i,"array"), np_array_test)
        np.save(os.path.join("data","01","dataset" + str(dataNumber),"test",i,"y"), y_test)

