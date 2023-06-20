import pandas as pd
from collections import Counter
import numpy as np
import pickle
import matplotlib.pyplot as plt
from random import sample
from math import ceil
import matplotlib.pyplot as plt

def CreateCluster(df):    
    cluster = {}
    for row in zip(df['sentence'], df['tags']):
        key = np.zeros((7,), dtype=int)    # count vector of of the entities name,quantity,unit,df,state,size,temp 
        counts_dict = Counter(row[1].split(" "))    # Dict of entity type vs frequency of that entity

        if 'NAME' in counts_dict:
            key[0] = counts_dict['NAME']

        if 'QUANTITY' in counts_dict:
            key[1] = counts_dict['QUANTITY']    

        if 'UNIT' in counts_dict:
            key[2] = counts_dict['UNIT']

        if 'DF' in counts_dict:
            key[3] = counts_dict['DF']

        if 'STATE' in counts_dict:
            key[4] = counts_dict['STATE']

        if 'SIZE' in counts_dict:
            key[5] = counts_dict['SIZE']

        if 'TEMP' in counts_dict:
            key[5] = counts_dict['TEMP']

        key = tuple(key)

        if key in cluster:
            cluster[key].append(row)
        else:
            _ = []
            _.append(row)
            cluster[key] = _
    return cluster

def PickleMe(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def LoadMe(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
        return obj
    
def FilterCluster(cluster):
    return dict(filter(lambda x:len(x[1]) >= 10, list(cluster.items())))

def Split25(cluster: dict):
    df = pd.DataFrame(columns=["sentence", "tags"])
    sample_ratio = 0.25
    for k,v in cluster.items():
        samples = sample(v, ceil(len(v)*sample_ratio))
        for item in samples:
            df.loc[len(df.index)] = [item[0], item[1]]
        
    return df

def SplitTrainTest(cluster: dict):
    df_train = pd.DataFrame(columns=["sentence", "tags"])
    df_test = pd.DataFrame(columns=["sentence", "tags"])
    sample_ratio = 0.1
    for k,v in cluster.items():
        samples = sample(v, ceil(len(v)*sample_ratio))
        train_samples = sample(samples, ceil(len(samples)*0.8))
        test_samples = list(set(samples) - set(train_samples))
        for item in train_samples:
            df_train.loc[len(df_train.index)] = [item[0], item[1]]
        for item in test_samples:
            df_test.loc[len(df_test.index)] = [item[0], item[1]]
        
    return df_train, df_test

def concat_dfs(df1, df2):
    return pd.concat([df1, df2])   

def SubtractDf(df1, df2):
    df = pd.merge(df1, df2, how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
    return df





    
if __name__ == "__main__":

    df = pd.read_csv("./LargeUnique.csv")
    df_small_test = pd.read_csv("./test.csv")
    df_pure = SubtractDf(df, df_small_test)
    print(len(df_pure))
    # print(len(df_small_test))
    # df_small_test_unique = df_small_test.drop_duplicates()
    # print(len(df_small_test_unique))
    cluster = CreateCluster(df_pure)
    PickleMe(cluster, "cluster")
    # print(f"number of clusters before filtering {len(cluster)}")
    # # for k,v in cluster.items():
    # #     print("cluster  " + str(k) + ":" + str(len(v)))

    # Not required as train test split is no longer needed. We are testing on gold labels of neerav
    # cluster_filtered = FilterCluster(cluster)
    # print(f"number of clusters after filtering {len(cluster_filtered)}")
    # # df_train, df_test = SplitTrainTest10(cluster)
    # df_train = Split25(cluster)
    # print("Raw 10% data")
    # print(df_train.head())
    # # print(df_test.head())
    # print(df_train.shape)
    # # print(df_test.shape)
    
    # # df_test.to_csv("SampleLargeTest.csv")
    # # df_sample_large_train = pd.read_csv("./SampleLargeTrain.csv")
    # df_train = df_train.iloc[:, 1:]
    # df_train.to_csv("SampleLarge25%.csv")
    # df_small_test = pd.read_csv("./test.csv")
    # df_small_test = df_small_test.iloc[:, 1:]
    # df_sample_large_train = SubtractDf(df_sample_large_train, df_small_test)
    # df_sample_large_train.to_csv("LargeSampleTrainPure.csv")
    # print("number of samples after removing gold samples:")
    # print(len(df_sample_large_train))
    # df_sample_large_train.drop_duplicates(inplace=True)
    # df_sample_large_train.to_csv("LargeSampleTrainPureWoDup.csv")
    # print("number of samples after removing gold samples and duplicates:")
    # print(len(df_sample_large_train))
    


    