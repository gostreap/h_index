from read_data import get_train_data, get_test_data
from fasttext_utils import small_class, df_to_txt, format_data
from utils import get_core_number, get_min_coauthor_hindex, get_max_coauthor_hindex, get_page_rank
import pandas as pd
import fasttext

def submit(model):
    train, _ = get_train_data()
    test, _ = get_test_data()

    l =[]
    for i in range (5):
        l.append(pd.read_csv("../tmp/data_test_part"+str(i+1)+".csv"))

    test = pd.concat(l)
    test.head()
    print(len(test))

    l =[]
    for i in range (18):
        l.append(pd.read_csv("../tmp/data_part"+str(i+1)+".csv"))

    train = pd.concat(l)
    train = small_class(train, 6)
    train.head()

    train_path = "../tmp/train.txt"
    df_to_txt(train,train_path)

    model_fasttext = fasttext.train_supervised(train_path,lr = 0.15815, dim = 2, epoch = 33, wordNgrams =3)

    train_core_number = get_core_number(train["author"])
    test_core_number = get_core_number(test["author"])

    train_min_coauthor_hindex = get_min_coauthor_hindex(train["author"])
    test_min_coauthor_hindex = get_min_coauthor_hindex(test["author"])

    train_max_coauthor_hindex = get_max_coauthor_hindex(train["author"])
    test_max_coauthor_hindex = get_max_coauthor_hindex(test["author"])

    train_pagerank = get_page_rank(train["author"])
    test_pagerank = get_page_rank(test["author"])

    X_train,y_train = format_data(train, model_fasttext, train_core_number, train_min_coauthor_hindex, train_max_coauthor_hindex, train_pagerank)
    X_test, y_test = format_data(test, model_fasttext, test_core_number, test_min_coauthor_hindex, test_max_coauthor_hindex, test_pagerank)
    

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test["hindex"] = y_pred
    submission = test[["author", "hindex"]]
    submission.to_csv("../tmp/submission.csv", index=None)