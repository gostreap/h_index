from pprint import pprint  # pretty-printer
from collections import defaultdict
import numpy as np
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pandas as pd
from gensim import corpora, models, similarities


def tokenize_abstracts():
    """Takes as input the column 'text' from the dataframe returned by get_processed data and saves a tokenized text"""
    docs = pd.read_csv("../tmp/processed_data.csv")["text"].to_list()

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
    lsi_model = models.LsiModel.load("../tmp/lsi")
    index = similarities.MatrixSimilarity.load('../tmp/gensim.index')
    loaded_corp = corpora.MmCorpus("../tmp/corpus")
    vec = np.zeros((217801,2*n))
    for i in tqdm(range(217801)):
        vec_lsi = lsi_model[loaded_corp[i]]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for j in range (n):
            vec[i,2*j] = sims[j+1][0]
            vec[i,2*j+1] = sims[j+1][1]

    data = pd.DataFrame(vec)
    data.to_csv("../tmp/similGraph.csv")
    return data