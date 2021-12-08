import json

import networkit as nk
import networkx as nx
import pandas as pd
from networkx.algorithms.cluster import triangles
from tqdm import tqdm

from read_data import (get_graph, get_nk_graph, get_train_data,
                       get_train_data_json)


def write_train_data_json():
    train, _ = get_train_data()
    train_data_json = {}
    for row in train.iterrows():
        train_data_json[str(row["author"])] = int(row["hindex"])
    with open("../data/train.json", "w") as f:
        json.dump(train_data_json, f)


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


def get_betweenness_centrality(author_ids):
    G, _, _ = get_graph()
    betweenness_centrality = nx.betweenness_centrality(G)
    author_betweenness_centrality = [
        betweenness_centrality[author_id] for author_id in author_ids
    ]
    df = pd.DataFrame(
        {"author": author_ids, "betweenness_centrality": author_betweenness_centrality}
    )
    return df


def get_triangles(author_ids):
    G, _, _ = get_graph()
    triangles = nx.algorithms.cluster.triangles(G)
    author_triangles = [triangles[author_id] for author_id in author_ids]
    df = pd.DataFrame({"author": author_ids, "triangles": author_triangles})
    return df


def get_clustering_coef(author_ids):
    G, _, _ = get_graph()
    clustering_coefs = nx.clustering(G, nodes=author_ids)
    author_clusering_coef = [clustering_coefs[author_id] for author_id in author_ids]
    df = pd.DataFrame({"author": author_ids, "clustering_coef": author_clusering_coef})
    return df


def get_eigenvector_centrality(author_ids):
    G, _, _ = get_graph()
    eigenvector_centralities = nx.algorithms.centrality.eigenvector_centrality(G)
    author_eigenvector_centrality = [
        eigenvector_centralities[author_id] for author_id in author_ids
    ]
    df = pd.DataFrame(
        {"author": author_ids, "eigenvector_centrality": author_eigenvector_centrality}
    )
    return df


def get_approx_closeness(author_ids, n_samples=2000):
    G, node_map = get_nk_graph()

    approx_closeness_model = nk.centrality.ApproxCloseness(G, n_samples)
    approx_closeness_model.run()

    approx_closeness = []
    for author_id in author_ids:
        approx_closeness.append(approx_closeness_model.score(node_map[str(author_id)]))

    df = pd.DataFrame({"author": author_ids, "approx_closeness": approx_closeness})
    return df


def get_closeness(author_ids):
    G, node_map = get_nk_graph()

    closeness_model = nk.centrality.Closeness(G, False, True)
    closeness_model.run()

    closeness = []
    for author_id in author_ids:
        closeness.append(closeness_model.score(node_map[str(author_id)]))

    df = pd.DataFrame({"author": author_ids, "closeness": closeness})
    return df


def get_harmonic(author_ids):
    G, node_map = get_nk_graph()

    harmonic_model = nk.centrality.HarmonicCloseness(G)
    harmonic_model.run()

    harmonic = []
    for author_id in author_ids:
        harmonic.append(harmonic_model.score(node_map[str(author_id)]))

    df = pd.DataFrame({"author": author_ids, "harmonic": harmonic})
    return df


def get_permanence(author_ids):
    G, node_map = get_nk_graph()

    permanence_model = nk.centrality.PermanenceCentrality(G, 100)
    permanence_model.run()

    permanence = []
    for author_id in author_ids:
        permanence.append(permanence_model.score(node_map[str(author_id)]))

    df = pd.DataFrame({"author": author_ids, "permanence": permanence})
    return df


def get_katz(author_ids):
    G, node_map = get_nk_graph()

    katz_model = nk.centrality.KatzCentrality(G, tol=1e-12)
    katz_model.run()

    katz = []
    for author_id in author_ids:
        katz.append(katz_model.score(node_map[str(author_id)]))

    df = pd.DataFrame({"author": author_ids, "katz": katz})
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


def get_mean_neighbors_degree(author_ids):
    G, _, _ = get_graph()
    mean_degree = []
    for author_id in tqdm(author_ids):
        neighbors = G.neighbors(author_id)
        total = 0
        for neighbor in neighbors:
            total += G.degree(neighbor)
        if len(list(neighbors)) != 0:
            mean_degree.append(total / len(list(neighbors)))
        else:
            mean_degree.append(0)
    df = pd.DataFrame({"author": author_ids, "mean_neighbors_degree": mean_degree})
    return df


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

    return pd.DataFrame(data)

