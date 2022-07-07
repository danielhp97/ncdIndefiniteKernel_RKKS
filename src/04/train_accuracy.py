import sys
from turtle import write
import yaml
import json
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

def metrics_calc(pd_dataset,params):
    metrics_Dict = {} # put everything in a dict
    label = pd_dataset.loc[0,'Class']
    # calculate acc
    metrics_Dict['acc'] = metrics.accuracy_score(y_true=pd_dataset['Class'], y_pred=pd_dataset['Prediction'])
    # calculate f1
    metrics_Dict['f1'] = metrics.f1_score(y_true=pd_dataset['Class'], y_pred=pd_dataset['Prediction'], pos_label=label)
    # calculate auc
    #fpr, tpr, thresholds = metrics.roc_curve(y_true=pd_dataset['class'], y_score=pd_dataset['Prediction'], pos_label=label)
    #metrics_Dict['auc'] = metrics.auc(fpr, tpr)
    return  metrics_Dict


if __name__== "__main__":
    i = sys.argv[1]
    i = str(i)
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    ncd_type = params['ncd_baseline']['ncd_type']

    base_dir = "data/03/dataset" + str(dataset_number) + "/"
    write_dir = "metrics/"
    # load results file
    results = pd.read_csv(base_dir + "{0}/results_test.csv".format(i))

    # calculate metrics
    final_metrics=metrics_calc(results,params)

    # write metrics to json
    with open(write_dir + 'metrics{0}.json'.format(i), 'w') as o:
        json.dump(final_metrics, o)

    