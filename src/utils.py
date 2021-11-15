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