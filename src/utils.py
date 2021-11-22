import networkx as nx
import csv
import json
from networkx.algorithms.cluster import clustering
from numpy import minimum
import pandas as pd
from read_data import get_graph, get_train_data_json
from tqdm import tqdm


def get_abstract_text(abstract):
    """If abstracts is the dictionnary return by read_data.get_abstracts() then this function take as input abstracts[paper_id]

    Returns:
        string: the text of the abstract
    """
    length = abstract["IndexLength"]
    text_tab = [None for _ in range(length)]
    for word, pos in abstract["InvertedIndex"].items():
        for i in pos:
            text_tab[i] = word
    return " ".join((filter((None).__ne__, text_tab)))


def get_coauthors_hindex(author_id, G, train_data_json):
    return [
        train_data_json[str(neighbor_id)]
        for neighbor_id in G.neighbors(author_id)
        if str(neighbor_id) in train_data_json
    ]

def get_number_of_coauthors_with_hindex(author_id, G, train_data_json):
    return len(get_coauthors_hindex(author_id, G, train_data_json))


def get_all_number_of_coauthors_with_hindex(authors_ids):
    G, _, _ = get_graph()
    train_data_json = get_train_data_json()
    n_coauthors = {}
    for author_id in tqdm(authors_ids):
        n_coauthors[author_id] = get_number_of_coauthors_with_hindex(
            author_id, G, train_data_json
        )
    return pd.DataFrame(
        list(n_coauthors.items()), columns=["author", "n_coauthors_with_hindex"]
    )


def get_core_number(author_ids):
    G, _, _ = get_graph()
    core_number = nx.core_number(G)
    author_core_numbers = [core_number[author_id] for author_id in author_ids]
    df = pd.DataFrame({"author": author_ids, "core_number": author_core_numbers})
    return df


def get_page_rank(author_ids):
    G, _, _ = get_graph()
    core_number = nx.pagerank(G)
    author_pagerank = [core_number[author_id] for author_id in author_ids]
    df = pd.DataFrame({"author": author_ids, "pagerank": author_pagerank})
    return df


def get_authority(author_ids):
    G, _, _ = get_graph()
    authority, _ = nx.hits(G)
    author_authority = [authority[author_id] for author_id in author_ids]
    df = pd.DataFrame({"author": author_ids, "authority": author_authority})
    return df


def get_clustering_coef(author_ids):
    G, _, _ = get_graph()
    clustering_coefs = nx.clustering(G, nodes=author_ids)
    author_clusering_coef = [clustering_coefs[author_id] for author_id in author_ids]
    df = pd.DataFrame({"author": author_ids, "clustering_coef": author_clusering_coef})
    return df


def get_hindex_info(author_ids, train_data_json):
    "Return the min, the mean and the max of the known hindex of the author in author_ids"
    hindexs = [
        train_data_json[str(author_id)]
        for author_id in author_ids
        if str(author_id) in train_data_json
    ]
    if len(hindexs) > 0:
        return (
            min(hindexs),
            sum(hindexs) / len(hindexs),
            max(hindexs),
        )
    else:
        return 1, 9.841160, 12


def get_neighborhood_info(author_ids, level=2):
    G, _, _ = get_graph()
    train_data_json = get_train_data_json()
    data = {"author": author_ids}
    for i in range(level):
        data["n_neighbors_dist_{}".format(i + 1)] = []
        data["min_neighbors_dist_{}".format(i + 1)] = []
        data["mean_neighbors_dist_{}".format(i + 1)] = []
        data["max_neighbors_dist_{}".format(i + 1)] = []

    for author_id in tqdm(author_ids):
        neighbors = set()
        for i in range(level):
            if i == 0:
                neighbors.update(list(G.neighbors(author_id)))
            else:
                for neighbor in neighbors.copy():
                    neighbors.update(list(G.neighbors(neighbor)))
            minimum, mean, maximum = get_hindex_info(neighbors, train_data_json)
            data["n_neighbors_dist_{}".format(i + 1)].append(len(neighbors))
            data["min_neighbors_dist_{}".format(i + 1)].append(minimum)
            data["mean_neighbors_dist_{}".format(i + 1)].append(mean)
            data["max_neighbors_dist_{}".format(i + 1)].append(maximum)

    print(len(data["author"]))
    print(len(data["n_neighbors_dist_{}".format(i + 1)]))
    return pd.DataFrame(data)
