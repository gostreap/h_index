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
      
    with open(csvFilePath, encoding='utf-8') as csvf: 
        csvReader = csv.DictReader(csvf) 
        for row in csvReader: 
            jsonArray.append(row)
  
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

csv_to_json("./data/train.csv", "./data/train.json")
csv_to_json("./data/test.csv", "./data/test.json")

def get_coauthorship_graph():
    """Constructs NetworX graph from input document with list of edges ./data/coauthorship.edgelist"""
    return nx.read_edgelist("./data/coauthorship.edgelist")

def get_coauthors_hindex(graph, n):
    """returns a list of all the h-indeces of the co-others of author wih id n"""

G = get_coauthorship_graph()
