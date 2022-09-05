import sys
from turtle import write
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def metrics_calc(pd_dataset,params):
    metrics_Dict = {} # put everything in a dict
    label = pd_dataset.loc[0,'Class']
    # calculate acc
    metrics_Dict['acc'] = metrics.accuracy_score(y_true=pd_dataset['Class'], y_pred=pd_dataset['Prediction'])
    # calculate f1
    metrics_Dict['f1'] = metrics.f1_score(y_true=pd_dataset['Class'], y_pred=pd_dataset['Prediction'], pos_label=label)
    # calculate auc
    pd_dataset['Prediction'] = [1 if item==label else 0 for item in pd_dataset['Prediction']]
    fpr, tpr, thresholds = metrics.roc_curve(pd_dataset['Class'], pd_dataset['Scores'], pos_label=label)
    metrics_Dict['auc'] = metrics.auc(fpr, tpr)
    # save fpr and tpr
    metrics_Dict['fpr'] = list(fpr)
    metrics_Dict['tpr'] = list(tpr)
    return  metrics_Dict


if __name__== "__main__":
    i = sys.argv[1]
    i = str(i)
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']

    base_dir = "data/03/dataset" + str(dataset_number) + "/"
    write_dir = "metrics/"
    # load results file
    results = pd.read_csv(base_dir + "{0}/results.csv".format(i))

    # calculate metrics
    final_metrics=metrics_calc(results,params)

    #write a auc graph to files

    # write metrics to json
    with open(write_dir + 'metrics{0}.json'.format(i), 'w') as o:
        json.dump(final_metrics, o)
    # write plots img do plots/auc{}.jpg
    plt.plot(final_metrics['fpr'], final_metrics['tpr'])
    plt.savefig('plots/auc{}.png'.format(i), bbox_inches='tight')
    plt.close()


    