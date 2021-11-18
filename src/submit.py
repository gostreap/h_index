from read_data import get_train_data, get_test_data
from fasttext_utils import small_class, df_to_txt, format_data
from utils import get_core_number
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import fasttext

def submit():
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

    model0 = fasttext.train_supervised(train_path,lr = 0.626905, dim = 12, epoch = 11, wordNgrams =3)

    train_core_number = get_core_number(train["author"])
    test_core_number = get_core_number(test["author"])

    X_train,y_train = format_data(train, model0, train_core_number)
    X_test, y_test = format_data(test, model0, test_core_number)

    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(X_train,y_train)

    y_pred = forest_model.predict(X_test)

    test["hindex"] = y_pred
    submission = test[["author", "hindex"]]
    submission.to_csv("../tmp/submission.csv", index=None)