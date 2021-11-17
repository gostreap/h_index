import json
import networkx as nx
import numpy as np
import pandas as pd


def get_train_data():
    df_train = pd.read_csv(
        "../data/train.csv", dtype={"author": np.int64, "hindex": np.float32}
    )
    n_train = df_train.shape[0]
    return df_train, n_train


def get_test_data():
    df_test = pd.read_csv("../data/test.csv", dtype={"author": np.int64})
    n_test = df_test.shape[0]
    return df_test, n_test


def get_graph():
    G = nx.read_edgelist("../data/coauthorship.edgelist", delimiter=" ", nodetype=int)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    return G, n_nodes, n_edges


def get_abstracts(n_lines=0):
    abstracts_file = open("./data/abstracts.txt", "r")
    abstracts = {}
    all_file = False
    if n_lines == 0:
        all_file == True
    for i, line in enumerate(abstracts_file):
        id, data = line.split("----")
        abstracts[int(id)] = json.loads(data)
        if not all_file and i == n_lines:
            return abstracts
    return abstracts


def get_author_papers():
    author_papers_file = open("../data/author_papers.txt", "r")
    author_papers = {}
    for line in author_papers_file:
        author_id, papers_string = line.split(":")
        papers_ids = papers_string.split("-")
        author_papers[int(author_id)] = [int(paper_id) for paper_id in papers_ids]
    return author_papers
