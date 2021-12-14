import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from IPython.display import clear_output


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    print("[%s] %s%s ...%s\r" % (bar, percents, "%", status))


def vectorizer(x, model, dic, n):
    progress(dic["i"] + 1, n)
    dic["i"] = dic["i"] + 1
    clear_output(wait=True)
    return model.infer_vector(str(x).split())


def add_do2vec_to_whole_dataset(dataset):
    """
    dataset: DataFrame of whole preprocessed data
    """
    print("loading data")
    whole_text = dataset["text"].apply(lambda x: str(x).split())

    def tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    data_for_training = list(tagged_document(whole_text))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=20, min_count=2, epochs=30)
    print("building vocab")
    model.build_vocab(data_for_training)
    print("training model")
    model.train(
        data_for_training, total_examples=model.corpus_count, epochs=model.epochs
    )
    print("saving model")
    model.save("doc2vec")
    print("vectorizing dataset")
    dic = {"i": 0}
    D2vec = dataset["text"].apply(lambda x: vectorizer(x, model, dic, dataset.shape[0]))
    D2vec2 = pd.DataFrame(list(D2vec), columns=["d2v" + str(i) for i in range(20)])
    D2vec2.index = dataset.index
    return pd.concat([dataset, D2vec2], axis=1)

