import csv
import json
import os

import fasttext
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from read_data import get_graph, get_test_data, get_train_data, get_train_data_json
from utils import (
    get_abstract_text,
    get_authority,
    get_clustering_coef,
    get_core_number,
    get_neighborhood_info,
    get_page_rank,
)


PROCESSED_DATA_PATH = "../tmp/processed_data.csv"
TRAIN_LENGTH = 174241


def normalize(X_train, X_test):
    m = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - m) / std
    X_test = (X_test - m) / std
    return X_train, X_test


def get_submission_data():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    data = data.drop(["author", "text", "modindx", "hindex_lab"], axis=1)
    train = data[:TRAIN_LENGTH]
    test = data[TRAIN_LENGTH:]
    X_train = train.drop("hindex", axis=1).to_numpy()
    y_train = train["hindex"].to_numpy()
    X_test = test.drop("hindex", axis=1).to_numpy()
    y_test = test["hindex"].to_numpy()
    return X_train, y_train, X_test, y_test


def get_numpy_data(n=10000):
    train = pd.read_csv(PROCESSED_DATA_PATH)[:TRAIN_LENGTH]
    train = train.sample(n=n, random_state=1)
    train, test = train_test_split(train, random_state=1)
    X_train = train.drop(
        ["author", "hindex", "text", "modindx", "hindex_lab"], axis=1
    ).to_numpy()
    y_train = train["hindex"].to_numpy()
    X_test = test.drop(
        ["author", "hindex", "text", "modindx", "hindex_lab"], axis=1
    ).to_numpy()
    y_test = test["hindex"].to_numpy()
    return X_train, y_train, X_test, y_test


def get_processed_data():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    return data[:TRAIN_LENGTH], data[TRAIN_LENGTH:]


def add_vectorized_text(data, model_fasttext):
    data = data.drop(
        [column for column in data.columns if column.startswith("vector_coord_")],
        axis=1,
    )
    vectors = data["text"].apply(
        lambda x: model_fasttext.get_sentence_vector(x)
        if not pd.isnull(x)
        else model_fasttext.get_sentence_vector("")
    )
    columns = ["vector_coord_{}".format(i) for i in range(len(vectors.iloc[0]))]
    vectors_df = pd.DataFrame(np.stack(vectors.to_numpy()), columns=columns)
    vectors_df["author"] = data["author"]
    return add_features(data, vectors_df)


def add_features(data, new_features):
    return data.merge(new_features, left_on="author", right_on="author", how="inner")


def clean_columns(data, neighborhood_level=2):
    valid_columns = [
        "author",
        "hindex",
        "text",
        "nb_paper",
        "core_number",
        "modindx",
        "hindex_lab",
        "n_coauthors_with_hindex",
        "pagerank",
        "authority",
        "clustering_coef",
    ]
    for i in range(neighborhood_level):
        valid_columns += [
                "n_neighbors_dist_{}".format(i + 1),
                "min_neighbors_dist_{}".format(i + 1),
                "mean_neighbors_dist_{}".format(i + 1),
                "max_neighbors_dist_{}".format(i + 1)
            ]
    valid_columns += [column for column in data if column.startswith("vector_coord_")]

    for column in data.columns:
        if column not in valid_columns:
            data = data.drop(column, axis=1)
    return data


def store_full_dataset_with_features(
    from_scratch=False, vectorize=True, neighborhood_level=2
):

    if from_scratch:
        train, _ = get_train_data()
        test, _ = get_test_data()
        test = test.drop("Unnamed: 0", axis=1)
        data = pd.concat([train, test], axis=0, ignore_index=True)

        store_whole_dataset(data, "../tmp/data")

        os.rename("../tmp/data_full.csv", PROCESSED_DATA_PATH)

    data = pd.read_csv(PROCESSED_DATA_PATH)

    data = clean_columns(data, neighborhood_level=neighborhood_level)

    print("Starting data columns :", list(data.columns))

    if not "core_number" in data.columns:
        print("Add core number to data")
        data = add_features(data, get_core_number(data["author"]))

    if not "pagerank" in data.columns:
        print("Add pagerank to data")
        data = add_features(data, get_page_rank(data["author"]))

    if not "authority" in data.columns:
        print("Add authority to data")
        data = add_features(data, get_authority(data["author"]))

    if not "clustering_coef" in data.columns:
        print("Add clustering coef to data")
        data = add_features(data, get_clustering_coef(data["author"]))

    if not "hindex_lab" in data.columns:
        print("Add small class to data")
        data = small_class(data, 6)

    if not "n_neighbors_dist_{}".format(neighborhood_level) in data.columns:
        print("Add neighborhood info to data")
        data = add_features(
            data, get_neighborhood_info(data["author"], level=neighborhood_level)
        )
    
    if vectorize:
        path_fasttext_text = "../tmp/fasttext_text.txt"
        df_to_txt(data[:TRAIN_LENGTH], path_fasttext_text)
        model_fasttext = fasttext.train_supervised(
            path_fasttext_text, lr=0.15815, dim=2, epoch=33, wordNgrams=3
        )
        os.remove(path_fasttext_text)
        data = add_vectorized_text(data, model_fasttext)

    print("Ending data columns :", list(data.columns))

    data.to_csv(PROCESSED_DATA_PATH, index=None)


def store_whole_dataset(data: pd.DataFrame, path: str):
    """
    Launch preprocessing_for_fastText by chunk of 10000 on the wall dataset passed in argument and stores it in a file.

    Args:
        data (pd.DataFrame): [description]
        path (string): [description]
    """
    start = 0
    end = 10000
    i = 1
    while start < data.shape[0]:
        if not end < data.shape[0]:
            print(end + 1)
            temp_data = preprocessing_for_fasttext(data, start, end + 1)
        else:
            print(end)
            temp_data = preprocessing_for_fasttext(data, start, end)
        temp_data.to_csv(path + str(i) + ".csv", index=None)
        start = end
        end = end + 10000 if end + 10000 <= data.shape[0] else data.shape[0]
        i = i + 1

    data_parts = []
    for i in range(1, i):
        part_path = path + str(i) + ".csv"
        data_parts.append(pd.read_csv(part_path))
        os.remove(part_path)
    data = pd.concat(data_parts)
    data.to_csv(path + "_full.csv", index=None)


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
    index = np.sort(np.array(data[:TRAIN_LENGTH]["hindex"].to_list())).reshape(-1, 1)
    clusters = KMeans(n_clusters=k, random_state=1).fit(index)
    data["modindx"] = data["hindex"].apply(lambda x: clusters.predict([[x]])[0] if not pd.isnull(x) else None)
    data["hindex_lab"] = data["modindx"].apply(lambda x: "__label__" + str(x) if not pd.isnull(x) else None)
    return data


def preprocessing_for_fasttext(data, start=0, end=0):
    # get a random subset of data
    sample_data = data.iloc[start:end, :]
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
        abstract_text_by_author_id[author_id] = ["", 0]

    df_abstract_text = pd.DataFrame.from_dict(
        abstract_text_by_author_id, orient="index", columns=["text", "nb_paper"]
    )
    df_abstract_text["author"] = df_abstract_text.index
    df_abstract_text = df_abstract_text.reset_index(drop=True)

    df_data = sample_data.merge(
        df_abstract_text, left_on="author", right_on="author", how="inner"
    )

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
