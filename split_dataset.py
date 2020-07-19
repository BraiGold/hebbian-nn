import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import subprocess
import sys
import os

import main

if __name__ == "__main__":

    data = pd.read_csv(f'./datasets/tp2_training_dataset.csv',header=None,dtype=np.double)
    X,categories = data.iloc[:,1:],data.iloc[:,0]

    for test_size in [10,20,30,40,50]:
        X_train, X_test, cat_train, cat_test = train_test_split(X, categories, test_size=test_size / 100, random_state=42,stratify=categories)
        train_data = pd.concat([cat_train,X_train],axis=1)
        test_data = pd.concat([cat_test,X_test],axis=1)
        data_file_train = f'ds_train_{100-test_size}_{test_size}.csv'
        data_file_test = f'ds_test_{100-test_size}_{test_size}.csv'
        train_data.to_csv(f'./datasets/{data_file_train}', header=False, index=False)
        test_data.to_csv(f'./datasets/{data_file_test}', header=False, index=False)

        if '-train' in sys.argv or '-retrain-all' in sys.argv:
            
            retrain = False
            if '-retrain-all' in sys.argv:
                retrain = True

            for rule in ['sanger','oja']:
                print(rule)
                print(f'train_size / test_size: {100 - test_size}/{test_size}')
                model_file = f'{rule}_{100-test_size}_{test_size}.sav'
                main.fit_if_necessary(model_file,data_file_train,rule=rule,retrain=retrain)
            
            