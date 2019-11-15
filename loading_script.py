import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
from IPython.display import display, clear_output
CSV_PATH = './datasets/train.csv'
CAST_NAME_REGEX = "(name\\':\ \\')([A-Z].+)(\\', \\'order)"
#hi its me


def get_movies_db():
    pd.set_option("display.max_colwidth", 10000)
    dataset = pd.read_csv(CSV_PATH)
    # print the number of rows in the data set
    number_of_rows = len(dataset)
    print("total samples: {}".format(number_of_rows))
    return dataset


def random_sample(dataset, samp_num):
    return dataset.sample(samp_num)

def get_column_list(df, column):
    col = df.iloc[:, column].to_list()
    return col


def dictioning_list(row):
    try:
        row = row[1:-2]
        list_of_cast = row.split('}, ')
        list_of_dict = []
        for i, cast_member in enumerate(list_of_cast):
            list_of_cast[i] = cast_member+'}'
            dic = ast.literal_eval(list_of_cast[i])
            list_of_dict.append(dic)
        return list_of_dict
    except (TypeError, SyntaxError):
        return []


def dictioning_column(column):
    dictioned_col = []
    count = 0
    for row in column:
        dictioned_col.append(dictioning_list(row))
        count += 1
    return dictioned_col




def init_genres_vocab(dataset):
    GENRES_VOCAB = {}
    count = 0
    col = dictioning_column(dataset["genres"])
    for row in col:
        for genre in row:
            if genre["name"] in GENRES_VOCAB:
                continue
            GENRES_VOCAB[genre["name"]] = count
            count += 1
    return GENRES_VOCAB

def init_cast_vocab(dataset):
    CAST_VOCAB = {}

    count = 0
    col = dictioning_column(dataset["cast"])
    for row in col:
        id_count = 0
        for cast_member in row:
            if id_count > 2:
                break
            id_count += 1
            if cast_member["name"] in CAST_VOCAB:
                continue
            CAST_VOCAB[cast_member["name"]] = count
            count += 1
    return CAST_VOCAB

def init_keyword_vocab(dataset):
    keywords_vocab = {}
    col = dictioning_column(dataset["Keywords"])
    count = 0
    for row in col:
        for keyword in row:
            if keyword["name"] in keywords_vocab:
                continue
            keywords_vocab[keyword["name"]] = count
            count += 1
    return keywords_vocab


def init_crew_vocab(dataset,job):
    crew_vocab = {}
    count = 0
    col = dictioning_column(dataset["crew"])
    for row in col:
        for crew_member in row:
            if crew_member["name"] in crew_vocab or crew_member["job"] not in job:
                continue
            crew_vocab[crew_member["name"]] = count
            count += 1
    return crew_vocab


PRODUCTION_THRESH = 10


def drop_small_prod(prod_dict):
    res_dict = {}
    count = 0
    for key in prod_dict:
        if prod_dict[key] > PRODUCTION_THRESH:
            res_dict[key] = count
            count += 1
    return res_dict


def init_prod_vocab(dataset):
    prod_vocab = {}
    col = dictioning_column(dataset["production_companies"])
    for row in col:
        for prod_comp in row:
            if prod_comp["name"] not in prod_vocab:
                prod_vocab[prod_comp["name"]] = 0
            prod_vocab[prod_comp["name"]] += 1
    return drop_small_prod(prod_vocab)






def test(dataset):


    GENRES_VOCAB = init_genres_vocab(dataset)
    CAST_VOCAB = init_cast_vocab(dataset)
    PRODUCER_VOCAB = init_crew_vocab(dataset, ['Director'])
    DIRECTOR_VOCAB = init_crew_vocab(dataset, ['Producer', 'Executive Producer'])
    KEYWORDS_VOCAB = init_keyword_vocab(dataset)
    PRODUCTION_COMPANY_VOCAB = init_prod_vocab(dataset)
    print(PRODUCTION_COMPANY_VOCAB)






def main():
    dataset = get_movies_db()
    test(dataset)



if __name__ == "__main__":
    main()

