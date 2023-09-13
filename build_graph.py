import os
import pandas as pd
import random
import numpy as np
import joblib
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
import re
from tqdm import tqdm

word_embeddings_dim = 300
word_vector_map = {}


doc_name_list = []
doc_train_list = []
doc_test_list = []

data = pd.read_csv('data.csv')


doc_name_list = data['ICD9_CODE'].values.tolist()
doc_content_list = data['TEXT'].values.tolist()

train_test_index = 0
for index in range(len(doc_name_list)):
    if index/len(doc_name_list) <= 0.9:
        train_test_index = index


train_ids = list(range(train_test_index))

random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/ICD.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = list(range(len(doc_name_list)))[train_test_index:]
# print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/ICD.test.index', 'w')
f.write(test_ids_str)
f.close()


ids = train_ids + test_ids


shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
shuffle_doc_words_str = re.sub(r"[^A-Za-z<>\n]"," ",shuffle_doc_words_str)

f = open('data/ICD_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/ICD_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

ICD_code_set = set() 
for ICD_code_words in shuffle_doc_name_list:
    words = ICD_code_words.strip().split(";")
    for word in words:
        ICD_code_set.add(word)
ICD_code_list = list(ICD_code_set) # 存储所有的ICD编码的列表
ICD_code_str = '\n'.join(ICD_code_list)

f = open('data/corpus/ICD_code.txt', 'w')
f.write(ICD_code_str)
f.close()


word_freq = {}
word_set = set()

stopwords = []
with open("stopwords.txt","r") as fp:
    for word in fp.readlines():
        word = re.sub(r"\s", "",word)
        stopwords.append(word)
    fp.close()

for doc_words in tqdm(shuffle_doc_words_list):
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    appeared = set()
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/ICD_vocab.txt', 'w')
f.write(vocab_str)
f.close()

definitions = []

for word in vocab:
    synsets = wn.synsets(word)
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)

f = open('data/corpus/ICD_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
# print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/ICD_word_vectors.txt', 'w')
f.write(string)
f.close()

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

word_vector_file = 'data/corpus/ICD_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])



label_set = set()
for doc_meta in shuffle_doc_name_list:
    label_set.add(doc_meta)
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/ICD_labels.txt', 'w')
f.write(label_list_str)
f.close()


train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
test_size = len(test_ids)

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/ICD.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()


row_x = []
col_x = []
data_x = []

for i in tqdm(range(real_train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    doc_len = len(words)
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))


f = open("data/ind.ICD.x", 'wb')
joblib.dump(x, f)
f.close()
del x

y = []

for i in tqdm(range(real_train_size)):
    doc_meta = shuffle_doc_name_list[i]
    codes = doc_meta.split(';')
    one_hot = [0 for l in range(len(ICD_code_list))]
    for code in codes:
        label_index = ICD_code_list.index(code)
        one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)


f = open("data/ind.ICD.y", 'wb')
joblib.dump(y, f)
f.close()
del y



row_tx = []
col_tx = []
data_tx = []

for i in tqdm(range(test_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    doc_len = len(words)
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))


f = open("data/ind.ICD.tx", 'wb')
joblib.dump(tx, f)
f.close()
del tx

ty = []
for i in tqdm(range(test_size)):

    doc_meta = shuffle_doc_name_list[i + train_size]
    codes = doc_meta.split(';')
    one_hot = [0 for l in range(len(ICD_code_list))]
    for code in codes:
        label_index = ICD_code_list.index(code)
        one_hot[label_index] = 1
    ty.append(one_hot)

ty = np.array(ty)


f = open("data/ind.ICD.ty", 'wb')
joblib.dump(ty, f)
f.close()
del ty

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in tqdm(range(train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    doc_len = len(words)
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))


f = open("data/ind.ICD.allx", 'wb')
joblib.dump(allx, f)
f.close()
del allx

ally = []

for i in tqdm(range(train_size)):

    doc_meta = shuffle_doc_name_list[i]
    codes = doc_meta.split(';')
    one_hot = [0 for l in range(len(ICD_code_list))]
    for code in codes:
        label_index = ICD_code_list.index(code)
        one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(ICD_code_list))]
    ally.append(one_hot)

ally = np.array(ally)


f = open("data/ind.ICD.ally", 'wb')
joblib.dump(ally, f)
f.close()
del ally

window_size = 5000
windows = []

for doc_words in tqdm(shuffle_doc_words_list):
    words_list = []
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    for word in words_list:
        if word in stopwords or len(word) <= 2:
            continue
        else:
            words_list.append(word)
    length = len(words_list)
    if length <= window_size:
        windows.append(words_list)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words_list[j: j + window_size]
            windows.append(window)
            # print(window)

word_window_freq = {}
for window in tqdm(windows):
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])


word_pair_count = {}
for window in tqdm(windows):
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []


num_window = len(windows)

for key in tqdm(word_pair_count):
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)


for i in tqdm(range(vocab_size)):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j) # 词向量相关性，范围[-1,1]
            if similarity > 1.4: # 词向量相关性高
                # print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)


doc_word_freq = {}

for doc_id in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[doc_id]
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    words = re.sub(r"[^A-Za-z]"," ",doc_words)
    words = words.strip().lower().split()
    doc_word_set = set()
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))


f = open("data/ind.ICD.adj", 'wb')
joblib.dump(adj, f)
f.close()
