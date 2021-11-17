import networkx as nx
import csv
import json


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


def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []

    with open(csvFilePath, encoding="utf-8") as csvf:
        csvReader = csv.DictReader(csvf)
        for row in csvReader:
            jsonArray.append(row)

    with open(jsonFilePath, "w", encoding="utf-8") as jsonf:
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)


def get_coauthors_hindex(author_id, G, train_data_json):
    return [
        train_data_json[str(neighbor_id)]
        for neighbor_id in G.neighbors(author_id)
        if str(neighbor_id) in train_data_json
    ]

def get_number_of_coauthors(author_id, G, train_data_json):
    return len(get_coauthors_hindex(author_id, G, train_data_json))

def get_coauthors_min_mean_max_hindex(author_id, G, train_data_json):
    coauthors_hindex = get_coauthors_hindex(author_id, G, train_data_json)
    if len(coauthors_hindex) > 0:
        return (
            min(coauthors_hindex),
            sum(coauthors_hindex) / len(coauthors_hindex),
            max(coauthors_hindex),
        )
    else:
        return None, None, None


def get_coauthors_mean_hindex(author_id, G, train_data_json):
    coauthors_hindex = get_coauthors_hindex(author_id, G, train_data_json)
    if len(coauthors_hindex) > 0:
        return sum(coauthors_hindex) / len(coauthors_hindex)
    else:
        return None


def get_all_coauthors_mean_hindex(authors_ids, G, train_data_json):
    hindex = {}
    for author_id in authors_ids:
        hindex[author_id] = get_coauthors_mean_hindex(author_id, G, train_data_json)
    return hindex


def get_all_number_of_coauthors(authors_ids, G, train_data_json):
    n_coauthors = {}
    for author_id in authors_ids:
        n_coauthors[author_id] = get_number_of_coauthors(author_id, G, train_data_json)
    return n_coauthors