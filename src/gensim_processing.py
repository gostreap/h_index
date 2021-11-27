from gensim_utils import tokenize_abstracts,latentSemanticIndexing,similarityIndex, get_author_abstract_similarity

dic, corp = tokenize_abstracts()

num_topics = 250
lsi = latentSemanticIndexing(dic,corp,num_topics)

index = similarityIndex(lsi,corp,num_topics)

data = get_author_abstract_similarity(10)