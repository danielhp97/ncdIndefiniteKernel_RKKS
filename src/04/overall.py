import sys
import os
from turtle import write
import yaml
import json
import numpy as np
import pandas as pd
from dagshub import dagshub_logger, DAGsHubLogger




if __name__== "__main__":
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    ncd_type = params['ncd_baseline']['ncd_type']

    base_dir = "data/03/dataset" + str(dataset_number) + "/"
    write_dir = "metrics/"
    # load results file
    json_list = [j for j in os.listdir("metrics/") if j[0]=='m']
    acc_mean = []
    f1_mean = []
    for js in json_list:
        json_path="metrics/" + str(js)
        with open(json_path) as j:
            data = json.load(j)
            acc_mean.append(data['acc'])
            f1_mean.append(data['f1'])
    # calculate metrics
    overall_json = {'Accuracy': np.mean(acc_mean),
                    'F1_Score': np.mean(f1_mean)}

    # write metrics to json
    with open('metrics/final_metrics.json', 'w') as o:
        json.dump(overall_json, o)

    # use dagshub logger
    logger = DAGsHubLogger()
    logger.log_metrics(overall_json)
    logger.log_hyperparams(params)


    