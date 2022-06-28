from cProfile import label
from pyexpat import model
import string
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import numpy as np
import pandas as pd
import pickle
from PIL import Image 
from learning_algorithms import  SquareHingeKreinSVM, kernel_matrix
from ncd import  Ncd


def data_preparation(base_dir, split="test"):
    data = pd.read_csv(base_dir + str(split) + "/" + str(i) + "/img_dump/labels.csv")
    name_dict = {}
    for u in data.iloc[:,1].unique():
        name_dict[u] = data[data.iloc[:,1]==u] # creates dict with x different datasets
    return name_dict

def compare_indices(generalDf, trainDf):
    index_list = []
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(generalDf['truePath'])
    for row in trainDf['truePath'].iteritems():
        print(row[1])
        indexes = generalDf.index[generalDf['truePath']==row[1]]
        print(indexes)
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

def train_model(ncd_kernel, x):
    model = SquareHingeKreinSVM()
    return model.fit(X = ncd_kernel, y = x)


if __name__ == "__main__":
    i = sys.argv[1]
    i = str(i)
    # train the model. we need to 
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    class1 = params['class1']
    class2 = params['class2']
    # what to do:

    # create separate lists according to the label of the image ()
    base_dir = "data/01/dataset" + str(dataset_number) + "/"
    data_dicts = data_preparation(base_dir) # outputs a dict with various datasets
    # we need to iterate each dataset
    # pegar no data/01/dataset/test/labels
    # kernel should be calculated here
    for key in data_dicts:
        # create new class instance for each data that exists
        # precisamos do ficheiro das labels,
        kernel = np.load("data/dataset{}/kernel/kernelmatrix.npy".format(dataset_number))
        label_dir = "data/01/dataset{}/train/{}/img_dump/".format(dataset_number, i)
        image_list, label_list = path_to_array(label_dir, class1, class2)
        # change class 1 to 1 and class 2 to -1
        label_list = [1 if i==class1 else -1 for i in label_list]
        label_list = np.array(label_list)
        # nova funcao, vai retornar uma lista com as imagens e outra com as labels
        indices = get_training_indices(kernel, 'data/dataset{0}/labels.csv'.format(dataset_number), class1, class2) #dictionary of training indices
        kernel_cut = Ncd.get_training_matrix(kernel, indices, dataset_number)
        model_instance = train_model(kernel_cut, label_list)
    
    # store images to be used later on other for testing
    dir_cleaning(i)
    #save model (is it needed? the class should be good across files)
    with open('data/02/dataset{0}/train/{1}/model_instances.pkl'.format(str(dataset_number), i), 'wb') as f:
        pickle.dump(model_instance, f)