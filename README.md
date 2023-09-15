TF-GCN
===
Combining Transformer-based Model and GCN to Predict ICD Codes from Clinical Records

How to use it?
------
Firstly, import the package.
```
from utils import *
from DL_ClassifierModel import *
```
1.How to preprocess the raw data
----
Instance the MIMIC-Ⅲ object and do preprocessing.
```
mimic = MIMIC_new(path='path to mimic3')
mimic.get_basic_data(outPath='data.csv')
```
After this, we can get a file named data.csv.

2.How to prepare the data class for the models.
---
Instance the data utils class and get the pretrained word embedding.
```
dataClass = (dataPath, mimicPath='mimic/', stopWordPath="stopwords.txt", validSize=0.2, testSize=0.0, minCount=10, noteMaxLen=768, seed=9527, topICD=-1)
```
3.How to compute ICD vectors.
---
Before train the model, we need to obtain the ICD vectors computed by ICDs' description first.
```
labDescVec = get_ICD_vectors(dataClass=dataClass, mimicPath="path to mimic3")
if dataClass.classNum=50:
    labDescVec = labDescVec[dataClass.icdIndex,:]
```
4.How to train the models.
---
Construct isomorphic images.
