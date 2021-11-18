import csv
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from gensim.utils import simple_preprocess
from read_data import get_graph, get_train_data_json

from utils import get_abstract_text, get_all_coauthors_mean_hindex, get_all_number_of_coauthors

def store_whole_dataset(train, path="../tmp/data_part"):
    # Store all data
    flag = True
    start = 0
    end = 0
    end = 10000 if end <= train.shape[0] else train.shape[0]
    i = 1
    while(end<= train.shape[0] and start< train.shape[0]):
        if not end < train.shape[0]:
            print(end+1)
            temp_data = preprocessing_for_fastText(0,train,start,end+1)
        else:
            print(end)
            temp_data = preprocessing_for_fastText(0,train,start,end)
        temp_data.to_csv(path+str(i)+".csv",index = None)
        start = end
        end = end+10000 if end+10000 <= train.shape[0] else train.shape[0]
        i = i+1

def df_to_txt(data, file_name):
    data[["hindex_lab", "text"]].to_csv(
        file_name,
        index=False,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ",
    )


def get_all_text_by_author_id(paper_id_to_author_id):
    ids = paper_id_to_author_id.keys()
    abstracts_file = open("../data/abstracts.txt", "r", encoding="utf-8")
    abstracts = {}
    for i, line in enumerate(abstracts_file):
        id, data = line.split("----", 1)
        if int(id) in ids:
            abstracts[paper_id_to_author_id[int(id)]] = (
                [json.loads(data)]
                if abstracts.get(paper_id_to_author_id[int(id)]) is None
                else [json.loads(data)] + abstracts.get(paper_id_to_author_id[int(id)])
            )
    return abstracts


def get_authors_id_by_papers_id_dict(ids):
    author_papers_file = open("../data/author_papers.txt", "r", encoding="utf-8")
    author_papers = {}
    for line in author_papers_file:
        author_id, papers_string = line.split(":")
        if int(author_id) in ids:
            papers_ids = papers_string.split("-")
            for paper_id in papers_ids:
                author_papers[int(paper_id)] = int(author_id)
    return author_papers


def small_class(data, k):
    index = np.sort(np.array(data['hindex'].to_list())).reshape(-1,1)
    clusters = KMeans(n_clusters=k,random_state=1).fit(index)
    data["modindx"] = data["hindex"].apply(lambda x: clusters.predict([[x]])[0])
    data["hindex_lab"] = data["modindx"].apply(lambda x: "__label__" + str(x))
    return data


def preprocessing_for_fastText(n_sample, data,start=0,end=0):
    # get a random subset of data
    if (end>start):
        sample_data = data.iloc[start:end,:]
    else:
        sample_data = data.sample(n=n_sample, random_state=1)
    author_ids = sample_data["author"].to_list()
    paper_id_to_author_id = get_authors_id_by_papers_id_dict(author_ids)
    
    abstract = get_all_text_by_author_id(paper_id_to_author_id)
    abstract_text_by_author_id = {}
    for author_id in abstract.keys():
        abstract_text_by_author_id[author_id] = [
            " ".join(
                simple_preprocess(
                    " ".join([get_abstract_text(r) for r in abstract[author_id]])
                )
            ),
            len(abstract[author_id]),
        ]

    for author_id in set(author_ids).difference(set(abstract.keys())):
        abstract_text_by_author_id[author_id]=["", 0]
    
    df_abstract_text = pd.DataFrame.from_dict(
        abstract_text_by_author_id, orient="index", columns=["text", "nb_paper"]
    )
    df_abstract_text["author"] = df_abstract_text.index
    df_abstract_text = df_abstract_text.reset_index(drop=True)

    df_data = sample_data.merge(
        df_abstract_text, left_on="author", right_on="author", how="inner"
    )

    G, _, _ = get_graph()
    train_data_json = get_train_data_json()
    coauthors_hindex = get_all_coauthors_mean_hindex(author_ids, G, train_data_json)
    n_coauthors = get_all_number_of_coauthors(author_ids, G, train_data_json)

    # TODO : change 9.841160 by mean value of hindex of author
    df_data["mean_coauthors_hindex"] = df_data["author"].apply(lambda author_id: coauthors_hindex[author_id] if coauthors_hindex[author_id] is not None else 9.841160)
    df_data["n_coauthors"] = df_data["author"].apply(lambda author_id: n_coauthors[author_id])

    return df_data


def general_comp(model, test):
    test_pred = model.predict(test["text"].to_list())
    test_pred_lab = [t[0] for t in test_pred[0]]
    test_pred_prob = [t[0] for t in test_pred[1]]
    df_test_pred = pd.DataFrame(
        {"test_pred_lab": test_pred_lab, "test_pred_prob": test_pred_prob}
    )
    test = test.reset_index(drop=True)
    test_comp = pd.concat([df_test_pred, test], axis=1)
    test_err = test_comp[test_comp["hindex_lab"] != test_comp["test_pred_lab"]]
    return test_err


def format_data(data, model, core_number):
    vectors = data["text"].apply(lambda x: model.get_sentence_vector(x) if not pd.isnull(x) else model.get_sentence_vector("")).to_list()    
    nb_data = np.array(data["nb_paper"].apply(lambda x: x if not pd.isnull(x) else 0).to_list()).reshape(-1, 1)
    coauthors_hindex_data = np.array(data["mean_coauthors_hindex"].to_list()).reshape(-1, 1)
    n_coauthors_data = np.array(data["n_coauthors"].to_list()).reshape(-1, 1)
    core_number = np.array(core_number["core_number"].to_list()).reshape(-1, 1)
    
    X = np.concatenate((vectors, nb_data, coauthors_hindex_data, n_coauthors_data, core_number), axis=1)
    y = np.array(data["hindex"].to_list())
    
    return X, y
