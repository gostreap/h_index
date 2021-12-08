# Data Challenge: H-index Prediction

Michael Fotso Fotso, Tristan François and Christian Kotait

## Setup

In order to execute our code, you must install the following dependencies.

```
catboost
fasttext
gensim
networkit
networkx
numpy
pandas
sklearn
scipy
```

You must also create a `data` and a `tmp` folder at the same level as `src`.

Finally, you must place all the initial data in the data folder. The following files are expected in the data folder:
```
abstracts.txt
author_papers.txt
coauthorship.edgelist
test.csv
train.csv
```

## Run

Then you just have to run the jupyter notebook `src/main.ipynb`.  Be careful, the data generation is long and requires a lot of resources.