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
    get_all_coauthors_mean_hindex,
    get_all_number_of_coauthors,
    get_all_number_of_coauthors_with_hindex,
    get_authority,
    get_core_number,
    get_max_coauthor_hindex,
    get_min_coauthor_hindex,
    get_all_number_of_second_degree_neighbors,
    get_page_rank,
)


PROCESSED_TRAIN_PATH = "../tmp/processed_train.csv"
PROCESSED_TEST_PATH = "../tmp/processed_test.csv"


def get_submission_data():
    train = pd.read_csv(PROCESSED_TRAIN_PATH)
    test = pd.read_csv(PROCESSED_TEST_PATH)
    X_train = train.drop(
        ["author", "hindex", "text", "modindx", "hindex_lab"], axis=1
    ).to_numpy()
    y_train = train["hindex"].to_numpy()
    X_test = test.drop(["author", "hindex", "text"], axis=1).to_numpy()
    y_test = test["hindex"].to_numpy()
    return X_train, y_train, X_test, y_test


def get_numpy_data(n=10000):
    train = pd.read_csv(PROCESSED_TRAIN_PATH)
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
    train = pd.read_csv(PROCESSED_TRAIN_PATH)
    test = pd.read_csv(PROCESSED_TEST_PATH)
    return train, test


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


def clean_columns(data):
    valid_columns = [
        "author",
        "hindex",
        "text",
        "nb_paper",
        "mean_coauthors_hindex",
        "n_coauthors",
        "core_number",
        "min_coauthor_hindex",
        "max_coauthor_hindex",
        "modindx",
        "hindex_lab",
        "n_coauthors_with_hindex",
        "pagerank",
        "authority",
        "n_neighbors_of_neighbors"
    ]
    valid_columns += [column for column in data if column.startswith("vector_coord_")]

    for column in data.columns:
        if column not in valid_columns:
            data = data.drop(column, axis=1)
    return data


def store_full_dataset_with_features(from_scratch=False, vectorize=True):

    if from_scratch:
        train, _ = get_train_data()
        test, _ = get_test_data()

        store_whole_dataset(train, "../tmp/train")
        store_whole_dataset(test, "../tmp/test")

        os.rename("../tmp/train_full.csv", PROCESSED_TRAIN_PATH)
        os.rename("../tmp/test_full.csv", PROCESSED_TEST_PATH)

    train = pd.read_csv(PROCESSED_TRAIN_PATH)
    test = pd.read_csv(PROCESSED_TEST_PATH)

    train = clean_columns(train)
    test = clean_columns(test)

    print("Starting train columns :", list(train.columns))
    print("Starting test columns :",list(test.columns))

    if not "core_number" in train.columns:
        print("Add core number to train")
        train = add_features(train, get_core_number(train["author"]))
    if not "core_number" in test.columns:
        print("Add core number to test")
        test = add_features(test, get_core_number(test["author"]))

    if not "min_coauthor_hindex" in train.columns:
        print("Add min coauthor hindex to train")
        train = add_features(train, get_min_coauthor_hindex(train["author"]))
    if not "min_coauthor_hindex" in test.columns:
        print("Add min coauthor hindex to test")
        test = add_features(test, get_min_coauthor_hindex(test["author"]))

    if not "max_coauthor_hindex" in train.columns:
        print("Add max coauthor hindex to train")
        train = add_features(train, get_max_coauthor_hindex(train["author"]))
    if not "max_coauthor_hindex" in test.columns:
        print("Add max coauthor hindex to test")
        test = add_features(test, get_max_coauthor_hindex(test["author"]))

    if not "pagerank" in train.columns:
        print("Add pagerank to train")
        train = add_features(train, get_page_rank(train["author"]))
    if not "pagerank" in test.columns:
        print("Add pagerank to test")
        test = add_features(test, get_page_rank(test["author"]))

    if not "n_coauthors_with_hindex" in train.columns:
        print("Add number of coauthors with hindex to train")
        train = add_features(
            train, get_all_number_of_coauthors_with_hindex(train["author"])
        )
    if not "n_coauthors_with_hindex" in test.columns:
        print("Add number of coauthors with hindex to test")
        test = add_features(
            test, get_all_number_of_coauthors_with_hindex(test["author"])
        )

    if not "authority" in train.columns:
        print("Add authority to train")
        train = add_features(train, get_authority(train["author"]))
    if not "authority" in test.columns:
        print("Add authority to test")
        test = add_features(test, get_authority(test["author"]))

    if not "n_neighbors_of_neighbors" in train.columns:
        print("Add number of neighbors of neighbors to train")
        train = add_features(
            train, get_all_number_of_second_degree_neighbors(train["author"])
        )
    if not "n_neighbors_of_neighbors" in test.columns:
        print("Add number of neighbors of neighbors to test")
        test = add_features(
            test, get_all_number_of_second_degree_neighbors(test["author"])
        )

    if not "hindex_lab" in train.columns:
        print("Add small class to train")
        train = small_class(train, 6)

    if vectorize:
        path_fasttext_text = "../tmp/fasttext_text.txt"
        df_to_txt(train, path_fasttext_text)
        model_fasttext = fasttext.train_supervised(
            path_fasttext_text, lr=0.15815, dim=2, epoch=33, wordNgrams=3
        )
        os.remove(path_fasttext_text)
        train = add_vectorized_text(train, model_fasttext)
        test = add_vectorized_text(test, model_fasttext)

    print("Ending train columns :", list(train.columns))
    print("Ending test columns :",list(test.columns))

    train.to_csv(PROCESSED_TRAIN_PATH, index=None)
    test.to_csv(PROCESSED_TEST_PATH, index=None)


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
            temp_data = preprocessing_for_fasttext(0, data, start, end + 1)
        else:
            print(end)
            temp_data = preprocessing_for_fasttext(0, data, start, end)
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
    index = np.sort(np.array(data["hindex"].to_list())).reshape(-1, 1)
    clusters = KMeans(n_clusters=k, random_state=1).fit(index)
    data["modindx"] = data["hindex"].apply(lambda x: clusters.predict([[x]])[0])
    data["hindex_lab"] = data["modindx"].apply(lambda x: "__label__" + str(x))
    return data


def preprocessing_for_fasttext(n_sample, data, start=0, end=0):
    # get a random subset of data
    if end > start:
        sample_data = data.iloc[start:end, :]
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
        abstract_text_by_author_id[author_id] = ["", 0]

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
    n_coauthors = get_all_number_of_coauthors(author_ids, G)

    # TODO : change 9.841160 by mean value of hindex of author
    df_data["mean_coauthors_hindex"] = df_data["author"].apply(
        lambda author_id: coauthors_hindex[author_id]
        if coauthors_hindex[author_id] is not None
        else 9.841160
    )
    df_data["n_coauthors"] = df_data["author"].apply(
        lambda author_id: n_coauthors[author_id]
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


def format_data(
    data, model, core_number, min_coauthors_hindex, max_coauthors_hindex, pagerank
):
    vectors = (
        data["text"]
        .apply(
            lambda x: model.get_sentence_vector(x)
            if not pd.isnull(x)
            else model.get_sentence_vector("")
        )
        .to_list()
    )
    nb_data = np.array(
        data["nb_paper"].apply(lambda x: x if not pd.isnull(x) else 0).to_list()
    ).reshape(-1, 1)
    coauthors_hindex_data = np.array(data["mean_coauthors_hindex"].to_list()).reshape(
        -1, 1
    )
    n_coauthors_data = np.array(data["n_coauthors"].to_list()).reshape(-1, 1)
    core_number = np.array(core_number["core_number"].to_list()).reshape(-1, 1)
    min_coauthors_hindex = np.array(
        min_coauthors_hindex["min_coauthor_hindex"].to_list()
    ).reshape(-1, 1)
    max_coauthors_hindex = np.array(
        max_coauthors_hindex["max_coauthor_hindex"].to_list()
    ).reshape(-1, 1)
    pagerank = np.array(pagerank["pagerank"].to_list()).reshape(-1, 1)

    X = np.concatenate(
        (
            vectors,
            nb_data,
            coauthors_hindex_data,
            n_coauthors_data,
            core_number,
            min_coauthors_hindex,
            max_coauthors_hindex,
            pagerank,
        ),
        axis=1,
    )
    y = np.array(data["hindex"].to_list())

    return X, y
