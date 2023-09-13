import pandas as pd
import re
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ICD():
    def __init__(self):
        self.load_csv()
        self.ICD_train_count_list = []
        self.ICD_train_count_dict = {}
        self.ICD_test_count_list = []
        self.ICD_test_count_dict = {}
        self.ICD_valid_count_list = []
        self.ICD_valid_count_dict = {}

    def get_data(self,M):
        for ICD_id in tqdm(self.train_id):
            self.ICD_count("train",self.ICD_lable_list[ICD_id])
        for ICD_id in tqdm(self.test_id):
            self.ICD_count("test",self.ICD_lable_list[ICD_id])
        for ICD_id in tqdm(self.valid_id):
            self.ICD_count("valid",self.ICD_lable_list[ICD_id])
        self.ICD_data_to_list()
        return self.ICD_train_count_list[0:M],self.ICD_test_count_list[0:M],self.ICD_valid_count_list[0:M]

    def load_csv(self,filename = 'data.csv'):
        data = pd.read_csv('data.csv')
        self.ICD_lable_list = data['ICD9_CODE'].values.tolist()
        self.train_valid_id,self.test_id = train_test_split(range(len(self.ICD_lable_list)), test_size=0.1, random_state=1013)
        self.train_id,self.valid_id = train_test_split(self.train_valid_id, train_size=0.9, random_state=1013)
    def ICD_count(self,icd_train_test,ICD_lable):
        result = re.split(";",ICD_lable)
        for icd in result:
            if icd_train_test == "train":
                if icd in self.ICD_train_count_dict.keys():
                    self.ICD_train_count_dict[icd] += 1
                else:
                    self.ICD_train_count_dict[icd] = 1
            elif icd_train_test == "test":
                if icd in self.ICD_test_count_dict.keys():
                    self.ICD_test_count_dict[icd] += 1
                else:
                    self.ICD_test_count_dict[icd] = 1
            elif icd_train_test == "valid":
                if icd in self.ICD_valid_count_dict.keys():
                    self.ICD_valid_count_dict[icd] += 1
                else:
                    self.ICD_valid_count_dict[icd] = 1

    def ICD_data_to_list(self):
        for tup in self.ICD_train_count_dict.items():
            self.ICD_train_count_list.append(tup)
        for tup in self.ICD_test_count_dict.items():
            self.ICD_test_count_list.append(tup)
        for tup in self.ICD_valid_count_dict.items():
            self.ICD_valid_count_list.append(tup)
        self.list_sort()
    def list_sort(self):
        def takeSecond(elem):
            return elem[1]
        self.ICD_train_count_list.sort(key=takeSecond,reverse=True)
        self.ICD_test_count_list.sort(key=takeSecond,reverse=True)
        self.ICD_valid_count_list.sort(key=takeSecond,reverse=True)


if __name__ == "__main__":
    #print(train_test_split(range(100), test_size=0.1, random_state=1013))
    ICD = ICD()
    data = ICD.get_data(20)
    print(data[0],data[1],data[2])
