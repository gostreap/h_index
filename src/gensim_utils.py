from pprint import pprint  # pretty-printer
from collections import defaultdict
import numpy as np
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pandas as pd
from gensim import corpora, models, similarities
import os

def create_dic_authordid_lineauthor(data):
    """returns a dictionnary linking the author id to the line of the author in the data dataframe"""
    dic = {}
    for index, row in data.iterrows():
        dic[str(index)] = row["author"]
    return dic

def tokenize_abstracts(docs):
    """Takes as input the column 'text' from the dataframe returned by get_processed data and saves a tokenized text"""
    docs = docs.to_list()
    # docs = pd.read_csv("../tmp/data10000.csv")["text"].to_list()
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    docs = [doc if not pd.isnull(doc) else "" for doc in tqdm(docs) ]
    for idx in tqdm(range(len(docs))):
        # print(docs[idx])
        if not pd.isnull(docs[idx]): 
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in tqdm(docs)]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in tqdm(docs)]

    # Remove common words
    stoplist = set('that for a of the and to in'.split())
    docs = [[token for token in doc if token not in stoplist] for doc in tqdm(docs)]

    # Remove words that appear only once
    frequency = defaultdict(int)
    for text in docs:
        for token in text:
            frequency[token] += 1

    docs = [
        [token for token in text if frequency[token] > 1]
        for text in docs
    ]
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]
    corpora.MmCorpus.serialize("../tmp/corpus", corpus)
    return dictionary,corpus


def latentSemanticIndexing(dictionary, corpus, number_topics):
    """returns and saves the latent semantic model of the corpus"""
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=number_topics)
    lsi.save("../tmp/lsi")
    return lsi

def similarityIndex(lsi_model,corpus,number_topics):
    """returns and save the similarity index of the corpus"""
    loaded_model = models.LsiModel.load("../tmp/lsi")
    index = similarities.Similarity("../tmp/index",lsi_model[corpus],num_features=number_topics)
    index.save('../tmp/gensim.index')
    return index


def get_author_abstract_similarity(n):
    """computes a dataframe containing the n closest authors and value of similarities to every author"""
    num_authors = 10000
    data = pd.read_csv("../tmp/data10000.csv")
    authorline_toID = create_dic_authordid_lineauthor(data)
    lsi_model = models.LsiModel.load(r"../tmp/lsi")
    index = similarities.MatrixSimilarity.load(r"../tmp/gensim.index")
    loaded_corp = corpora.MmCorpus(r"../tmp/corpus")
    
    temp_data =np.zeros((num_authors,2*n))
    for i in tqdm(range(num_authors)):
        vec_lsi = lsi_model[loaded_corp[i]]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for j in range (n):
            temp_data[i,2*j] = authorline_toID[str(sims[j+1][0])]
            temp_data[i,2*j+1] = sims[j+1][1]
    temp_data = pd.DataFrame(temp_data)
    temp_data.insert(0,"author",pd.read_csv("../tmp/data10000.csv")["author"].to_list())
    temp_data.to_csv("../tmp/similGraph_full.csv", index=None)
    # start = 0
    # end = 1000
    # pas = 1000
    # ind = 0
    # while start < num_authors:
    #     temp_data =np.zeros((pas,2*n))
    #     for i in tqdm(range(start,end)):
    #         vec_lsi = lsi_model[loaded_corp[i]]
    #         sims = index[vec_lsi]
    #         sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #         for j in range (n):
    #             temp_data[i%pas,2*j] = sims[j+1][0]
    #             temp_data[i%pas,2*j+1] = sims[j+1][1]
    #     pd.DataFrame(temp_data).to_csv("../tmp/similGraph" + str(ind) + ".csv", index=None)
    #     start = end
    #     end = end + pas if end + pas <= num_authors else num_authors
    #     ind+=1
    
    # data_parts = []
    # for i in range(0, ind):
    #     part_path = "../tmp/similGraph"  + str(i) + ".csv"
    #     data_parts.append(pd.read_csv(part_path))
    #     os.remove(part_path)
    # data = pd.concat(data_parts)
    # data.insert(0,"author",pd.read_csv("../tmp/data_10000.csv")["author"].to_list())
    # data.to_csv("../tmp/similGraph"  + "_full.csv", index=None)
