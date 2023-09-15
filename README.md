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
