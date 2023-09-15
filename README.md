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
