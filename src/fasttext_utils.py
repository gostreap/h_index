import csv
import json
import pandas as pd

from gensim.utils import simple_preprocess

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


def getText(dic):
    temp = [""] * dic["IndexLength"]
    for word, pos in dic["InvertedIndex"].items():
        for i in pos:
            temp[i] = word
    return " ".join((filter((None).__ne__, temp)))


def get_abstracts_by_dic(dic):
    ids = dic.keys()
    abstracts_file = open("../data/abstracts.txt", "r", encoding="utf-8")
    abstracts = {}
    for i, line in enumerate(abstracts_file):
        id, data = line.split("----", 1)
        if int(id) in ids:
            abstracts[dic[int(id)]] = (
                [json.loads(data)]
                if abstracts.get(dic[int(id)]) is None
                else [json.loads(data)] + abstracts.get(dic[int(id)])
            )
    return abstracts


def get_authors_papers_id_by_ids(ids):
    author_papers_file = open("../data/author_papers.txt", "r", encoding="utf-8")
    author_papers = {}
    for line in author_papers_file:
        author_id, papers_string = line.split(":")
        if int(author_id) in ids:
            papers_ids = papers_string.split("-")
            for paper_id in papers_ids:
                author_papers[int(paper_id)] = int(author_id)
    return author_papers


def preprocessing_for_fastText(n_sample, data):
    data1 = data.sample(n=n_sample, random_state=1)
    ids = data1["author"].to_list()
    paper_id_dic = get_authors_papers_id_by_ids(ids)
    abstract = get_abstracts_by_dic(paper_id_dic)
    abstract_text = {}
    for id in abstract.keys():
        abstract_text[id] = " ".join(
            simple_preprocess(
                " ".join([" ".join(r["InvertedIndex"]) for r in abstract[id]])
            )
        )
    data1["hindex_lab"] = data1["hindex"].apply(lambda x: "__label__" + str(x))
    df_abstract_text = pd.DataFrame.from_dict(
        abstract_text, orient="index", columns=["text"]
    )
    df_abstract_text["author"] = df_abstract_text.index
    df_abstract_text = df_abstract_text.reset_index(drop=True)
    df_data = data1.merge(
        df_abstract_text, left_on="author", right_on="author", how="inner"
    )
    return df_data


def small_class(data, k):
    maxi = data["hindex"].max()
    data["modindx"] = data["hindex"].apply(lambda x: x if x < k else k)
    data["hindex_lab"] = data["modindx"].apply(lambda x: "__label__" + str(x))
    return data


def preprocessing2_for_fastText(n_sample, data):
    data1 = data.sample(n=n_sample, random_state=1)
    ids = data1["author"].to_list()
    paper_id_dic = get_authors_papers_id_by_ids(ids)
    abstract = get_abstracts_by_dic(paper_id_dic)
    abstract_text = {}
    for id in abstract.keys():
        abstract_text[id] = [
            " ".join(simple_preprocess(" ".join([getText(r) for r in abstract[id]]))),
            len(abstract[id]),
        ]
    # data1['hindex_lab'] = data1['modindx'].apply(lambda x:'__label__'+str(x))
    df_abstract_text = pd.DataFrame.from_dict(
        abstract_text, orient="index", columns=["text", "nb_paper"]
    )
    df_abstract_text["author"] = df_abstract_text.index
    df_abstract_text = df_abstract_text.reset_index(drop=True)
    df_data = data1.merge(
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