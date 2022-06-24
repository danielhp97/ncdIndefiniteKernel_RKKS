from logging import raiseExceptions
import os
import re
import csv
import yaml
import pandas as pd
from sklearn.model_selection import KFold
# prepare datasets: labelling and division

#open parameter file
with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
params=params['Parameters']
dataset_number = params['dataset']

#get dataset to work on
datasetNumber= params['dataset']

def label_dataset1():
    all = os.listdir('data/dataset' + str(dataset_number))
    categories = []
    for a in all:
        if os.path.isdir(os.path.join('data/dataset' + str(dataset_number),a)):
            if a not in ['train', 'test', 'validation']:
                categories.append(a)
    #for each category: get list of pictures, add label 'category name' and picture path
    os.remove(os.path.join("data","dataset"+str(dataset_number),"labels.csv"))
    for c in categories:
        imgs = os.listdir('data/dataset' + str(dataset_number) + '/' + c +'/')
        for i in imgs:
            print(i)
            entry = []
            if i == "compressedImages.npy":
                pass
            elif i == "kernelmatrix.npy":
                pass
            elif i == "labels.csv":
                pass
            elif i == ".DS_Store":
                pass
            else:
                print(i)
                number = re.search("[1-9]+", i).group(0) #regex change name of picture (to label)
                name = c + '_' + number + '.jpg' #regex change name of picture (to label)
                path = '/dataset' + str(dataset_number) + '/' + c + '/' + i
                entry.append((name, c, path)) # accordion_0017.jpg | accordion | /accordion/Image_0017.jpg
            # remove labels file
            with open('data/dataset' + str(dataset_number) + '/labels.csv', 'a', newline='') as f:
                writer_obj = csv.writer(f)
                writer_obj.writerow(entry)
                f.close() #write into csv
            df = pd.read_csv('data/dataset' + str(dataset_number) + '/labels.csv')
            df.to_csv('data/dataset' + str(dataset_number) + '/labels.csv', index=False)        

def dataset1_subset():
    labels = pd.read_csv('data/dataset' + str(dataset_number) + '/labels.csv', header=None, quotechar="'", sep=',')
    labels0_fixed = labels.iloc[:,0].str.split("(",expand=True)[1]
    labels2_fixed = labels.iloc[:,2].str.split(")",expand=True)[0]
    labels.iloc[:,0] = labels0_fixed
    labels.iloc[:,2] = labels2_fixed
    if params['multiclassification']:
        print('Does not work - Necessary to make several binary problems')
        raiseExceptions('Multiclassification is not currently available')
    else:
        # get index as well here, we need the index of the subsets 
        labels_list = labels.index[((labels.iloc[:,1].str.contains(params['class1'])) | (labels.iloc[:,1].str.contains(params['class2']))) == True].tolist()
        f_labeled = labels[(labels.iloc[:,1].str.contains(params['class1'])) | (labels.iloc[:,1].str.contains(params['class2']))] #subset dataset
    return(f_labeled,labels_list)

def dataset1_kfoldsplit():
    # cross validation
    KF = KFold(n_splits=params['kfold_nr'], shuffle=True) #different sets created
    # we need to iterate over each set and store the index
    i=0
    dtrain_x={}; dtrain_y={} # dics with train data datapath
    dtest_x={}; dtest_y={} # dics with test data datapath
    dtrain_indexes = {}; dtest_indexes = {} # dict for index storing
    for train_index, test_index in KF.split(dataset.iloc[:,2]): 
        # stored path for each image in dics
        dtrain_indexes["{0}".format(i)] = train_index
        dtest_indexes["{0}".format(i)] = test_index
        dtrain_x["{0}".format(i)], dtest_x["{0}".format(i)] = dataset.iloc[train_index,2], dataset.iloc[test_index,2] 
        dtrain_y["{0}".format(i)], dtest_y["{0}".format(i)] = dataset.iloc[train_index,1], dataset.iloc[test_index,1]
        i+=1
    return dtrain_x, dtrain_y, dtest_x, dtest_y, dtrain_indexes, dtest_indexes

def dataset1_dir_cleaning(x):
    if os.path.isfile("data/00/dataset" + str(dataset_number) + "/train/{0}/x.csv".format(x)): # check if labels exist: if yes, delete it.
            os.remove("data/00/dataset" + str(dataset_number) + "/train/{0}/x.csv".format(x))
    if os.path.isfile("data/00/dataset" + str(dataset_number) + "/train/{0}/y.csv".format(x)): # check if labels exist: if yes, delete it.
            os.remove("data/00/dataset" + str(dataset_number) + "/train/{0}/y.csv".format(x))
    if os.path.isfile("data/00/dataset" + str(dataset_number) + "/test/{0}/x.csv".format(x)): # check if labels exist: if yes, delete it.
            os.remove("data/00/dataset" + str(dataset_number) + "/test/{0}/x.csv".format(x))
    if os.path.isfile("data/00/dataset" + str(dataset_number) + "/test/{0}/y.csv".format(x)): # check if labels exist: if yes, delete it.
            os.remove("data/00/dataset" + str(dataset_number) + "/test/{0}/y.csv".format(x))
    if os.path.exists("data/00/dataset" + str(dataset_number) + "/train/{0}".format(x)): # check if folder 0/4 exists, if yes, delete it.
            os.rmdir("data/00/dataset" + str(dataset_number) + "/train/{0}/".format(x))
    if os.path.exists("data/00/dataset" + str(dataset_number) + "/test/{0}/".format(x)): # check if folder 0/4 exists, if yes, delete it.
            os.rmdir("data/00/dataset" + str(dataset_number) + "/test/{0}/".format(x))
    if not os.path.exists("data/00/dataset" + str(dataset_number) + "/test/"): # check if train folder:
        os.mkdir("data/00/dataset" + str(dataset_number) + "/test/")
    if not os.path.exists("data/00/dataset" + str(dataset_number) + "/train/"): # check if train folder:
        os.mkdir("data/00/dataset" + str(dataset_number) + "/train/")
    os.mkdir("data/00/dataset" + str(dataset_number) + "/train/{0}".format(x))
    os.mkdir("data/00/dataset" + str(dataset_number) + "/test/{0}".format(x))

def dataset1_split():
    dtrain_x, dtrain_y, dtest_x, dtest_y, dtrain_indexes, dtest_indexes = dataset1_kfoldsplit()
    for i in range(0,params['kfold_nr']):
        dataset1_dir_cleaning(i) #check if directories and csv exist
        # export to csv
        dtrain_x["{0}".format(i)].to_csv(path_or_buf="data/00/dataset" + str(dataset_number) + "/train/{0}/x.csv".format(i),index=False,header=False)
        dtrain_y["{0}".format(i)].to_csv(path_or_buf="data/00/dataset" + str(dataset_number) + "/train/{0}/y.csv".format(i),index=False,header=False)
        dtest_x["{0}".format(i)].to_csv(path_or_buf="data/00/dataset" + str(dataset_number) + "/test/{0}/x.csv".format(i),index=False,header=False)
        dtest_y["{0}".format(i)].to_csv(path_or_buf="data/00/dataset" + str(dataset_number) + "/test/{0}/y.csv".format(i),index=False,header=False)
        # np.save("") dindexes in ../img_dump
        # np.save("") dindexes test in ../img_dump


if params['preprocess']['label']:
    label_dataset1()
    
if params['preprocess']['subset']:
    dataset, label_list = dataset1_subset()

if params['preprocess']['split']:
    dataset1_split()