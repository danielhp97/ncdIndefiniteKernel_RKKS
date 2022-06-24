from copyreg import pickle
import pickle
from pathlib import Path
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import time
import numpy as np
import pandas as pd
from PIL import Image
from functools import partial
try:
    import lzma
except ImportError:
    from backports import lzma
from ncd import Ncd

################################################################################

def sign(x, class1, class2):
    value = np.sign(x)
    if value == -1:
        value = class2
    else:
        value = class1
    return value

def data_preparation(base_dir, split="test"):
    data = pd.read_csv(base_dir + str(split) + "/" + str(i) + "/img_dump/labels.csv")
    return data
        
def ncd_preparation(dataset, i):
    data = dataset
    np_images = []
    temp = data['File']
    for index, row in data.iterrows():
        row['Path'] = 'data/01/dataset{}/test/{}/img_dump/{}'.format(dataset_number,i ,row['File'])
    path_list = data['Path']
    counter_x = 0
    for i in path_list:
        start_time = time.time()
        print("imagem x: {0}".format(counter_x))
        temp = Image.open(i)
        temp = np.asanyarray(temp, dtype=object)
        np_images.append(temp)
        counter_x +=1
        print("Tempo de execucao: {}".format(time.time()-start_time))
    return np_images

def path_to_array(label_dir, class1, class2):
    df = pd.read_csv(label_dir + 'labels.csv')
    image_list = []
    for i in df.iloc[:,0]:
        temp = Image.open(label_dir + str(i))
        temp = np.asanyarray(temp, dtype=object)
        image_list.append(temp)
    df.iloc[:,1].replace({str(class1): 1, str(class2): -1}) # careful with the extra '' that are on the label
    label_list = df.iloc[:,1].to_list()  
    return image_list, label_list


def compare_indices(generalDf, trainDf):
    index_list = []
    for row in trainDf['truePath'].iteritems():
        indexes = generalDf.index[generalDf['truePath']==row[1]]
        index_list.append(indexes[0])
    return(index_list)
        
def get_training_indices(kernel, general_labels_path, class1, class2):
    # we need to get a smaller matrix, taking into account the size of the train image vector
    kernel = kernel # load matrix
    # how to check indices of matrix:
    labels= pd.read_csv(general_labels_path, sep = ", ")# open general labels.csv
    # limpar o dataset
    labels = labels[(labels.iloc[:,1].str.contains(class1)) | (labels.iloc[:,1].str.contains(class2))]# filter t by classes name
    labels = labels.reset_index()
    labels = labels.drop(['index'],axis = 1)
    labels.columns = ['image', 'label', 'truePath']
    labels['truePath'] = labels['truePath'].str.slice(1,-1)
    labels['truePath'] = labels['truePath'].str.slice(0,-2)
    labels['truePath'] = 'data' + labels['truePath']
    #labels['truePath'] = labels.truePath.str(-2)
    labels_train = pd.read_csv("data/01/dataset{0}/train/{1}/img_dump/labels.csv".format(dataset_number, i))# open as dataframe iteration train labels
    labels_train.columns = ['image', 'label']
    labels_train['label'] = labels_train['label'].str.slice(2,-1)
    labels_train["truePath"] = "data/dataset" + str(dataset_number) + "/" + labels_train['label'] + "/" + labels_train['image']
    # taking into account column 1 (image_0000.jpg) and 2 (label) create an data/dataset{}/{nlabel}/{image_name}
    # for each element in that line, check the corresponding index of that file in the general labels file.
    indexes = compare_indices(labels, labels_train)
    # append that index to a list
    return indexes

def get_testing_indices(kernel, general_labels_path, class1, class2):
    # we need to get a smaller matrix, taking into account the size of the train image vector
    kernel = kernel # load matrix
    # how to check indices of matrix:
    labels= pd.read_csv(general_labels_path, sep = ", ")# open general labels.csv
    # limpar o dataset
    labels = labels[(labels.iloc[:,1].str.contains(class1)) | (labels.iloc[:,1].str.contains(class2))]# filter t by classes name
    labels = labels.reset_index()
    labels = labels.drop(['index'],axis = 1)
    labels.columns = ['image', 'label', 'truePath']
    labels['truePath'] = labels['truePath'].str.slice(1,-1)
    labels['truePath'] = labels['truePath'].str.slice(0,-2)
    labels['truePath'] = 'data' + labels['truePath']
    #labels['truePath'] = labels.truePath.str(-2)
    labels_test = pd.read_csv("data/01/dataset{0}/test/{1}/img_dump/labels.csv".format(dataset_number, i))# open as dataframe iteration train labels
    labels_test.columns = ['image', 'label']
    labels_test['label'] = labels_test['label'].str.slice(2,-1)
    labels_test["truePath"] = "data/dataset" + str(dataset_number) + "/" + labels_test['label'] + "/" + labels_test['image']
    # taking into account column 1 (image_0000.jpg) and 2 (label) create an data/dataset{}/{nlabel}/{image_name}
    # for each element in that line, check the corresponding index of that file in the general labels file.
    indexes = compare_indices(labels, labels_test)
    # append that index to a list
    return indexes

################################################################################

def dir_cleaning(x):
    #deleting
    if os.path.exists("data/03/dataset" + str(dataset_number) + "/{0}".format(x)):
        for f in os.path.os.listdir("data/03/dataset" + str(dataset_number) + "/{0}".format(x)):
            # ver se e directoria
            if os.path.isdir(os.path.join("data/03/dataset" + str(dataset_number) + "/{0}".format(x),f)):
                for f2 in os.path.os.listdir("data/03/dataset" + str(dataset_number) + "/{0}/{1}".format(x,f)):
                    os.remove(os.path.join("data/03/dataset" + str(dataset_number) + "/{0}/{1}".format(x,f), f2))
                os.rmdir(os.path.join("data/03/dataset" + str(dataset_number) + "/{0}".format(x),f))
            elif os.path.isfile(os.path.join("data/03/dataset" + str(dataset_number) + "/{0}".format(x),f)):
                os.remove(os.path.join("data/03/dataset" + str(dataset_number) + "/{0}".format(x),f))
        os.rmdir("data/03/dataset" + str(dataset_number) + "/{0}".format(x))
    #### writing
    os.mkdir("data/03/dataset" + str(dataset_number) + "/{0}".format(x))


if __name__ == "__main__":
    i = sys.argv[1]
    i = str(i)
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    ncd_type = params['ncd_baseline']['ncd_type']
    class1 = params['class1']
    class2 = params['class2']
    
    # iterate each image and append to a dataframe the predicted category
    base_dirtest = "data/01/dataset" + str(dataset_number) + "/"
    base_dir = "data/02/dataset" + str(dataset_number) + "/"
    data = data_preparation(base_dirtest)
    data.columns = ['File', 'Class']
    data['Path'] = ""
    temp = data['Class']
    temp = [t[2:] for t in temp]
    temp = [t[:-1] for t in temp]
    #temp = temp[-1:]
    data['Class']=temp
    data_array = ncd_preparation(data, i)
    # labels
    label_dir = "data/01/dataset{}/test/{}/img_dump/".format(dataset_number, i)
    image_list, label_list = path_to_array(label_dir, 'kangaroo', 'pigeon')
    kernel = np.load("data/dataset{}/kernel/kernelmatrix.npy".format(dataset_number))
    indices_train = get_training_indices(kernel, 'data/dataset{0}/labels.csv'.format(dataset_number), class1, class2) #dictionary of train indices
    indices_test = get_testing_indices(kernel, 'data/dataset{0}/labels.csv'.format(dataset_number), class1, class2) #dictionary of test indices 
    kernel_cut = Ncd.get_testing_matrix(kernel, indices_train, indices_test, dataset_number)
    # prediction
    #import model
    with open('data/02/dataset{0}/train/{1}/model_instances.pkl'.format(str(dataset_number), i), "rb") as input_file:
        model = pickle.load(input_file)
    predictions = []
    true_labels = data['Class']
    predictions = model.predict(kernel_cut)
    # create dataframe with data original values and predictions
    final_data = data
    
    final_data.columns=['Prediction', 'Class', 'Path']
    final_data['Prediction'] = predictions
    final_data = pd.DataFrame(final_data, columns = ['Prediction', 'Class', 'Path'])
    final_data['Prediction'] = [ sign(x, class1, class2) for x in final_data['Prediction']]

    # store results
    dir_cleaning(i)
    print("Storing data on disk")
    final_data.to_csv("data/03/dataset{0}/{1}/results.csv".format(dataset_number, i))