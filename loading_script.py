import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
from IPython.display import display, clear_output


from datetime import datetime
import logging

from boosting import *

CSV_PATH = './train.csv'
CAST_NAME_REGEX = "(name\\':\ \\')([A-Z].+)(\\', \\'order)"



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
    return col, GENRES_VOCAB

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
    return col, CAST_VOCAB

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
    return col, keywords_vocab


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
    return col, crew_vocab


PRODUCTION_THRESH = 10


def drop_small_prod(prod_dict, thresh):
    res_dict = {}
    count = 0
    for key in prod_dict:
        if prod_dict[key] > thresh:
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
    return col, drop_small_prod(prod_vocab, PRODUCTION_THRESH),



COUNTRY_THRESH = 5

def init_production_countries_vocab(dataset):
    countries_vocab = {}
    col = dictioning_column(dataset["production_countries"])
    for row in col:
        for prod_country in row:
            if prod_country["name"] not in countries_vocab:
                countries_vocab[prod_country["name"]] = 0
            countries_vocab[prod_country["name"]] += 1
    return col, drop_small_prod(countries_vocab, COUNTRY_THRESH)

LANGUAGE_THRESH = 10

def init_language_vocab(dataset):
    language_vocab = {}
    col = dictioning_column(dataset["spoken_languages"])
    for row in col:
        for prod_country in row:
            if prod_country["name"] not in language_vocab:
                language_vocab[prod_country["name"]] = 0
            language_vocab[prod_country["name"]] += 1
    return col, drop_small_prod(language_vocab, LANGUAGE_THRESH)

def extract_release_year_month(column):
    year_column = []
    month_column = []
    year_column = [datetime.strptime(x, '%m/%d/%y') for x in column]
    for i, x in enumerate(year_column):
        if x.year > 2030:
            x = x.replace(year=x.year-100)
        year_column[i] = x.strftime('%Y')

        month_column.append(x.strftime('%m'))
    return year_column, month_column


def build_rep_vec_row(row, vocab_dict):
    rep_vec = np.zeros(len(vocab_dict), dtype=int)
    for item in row:
        if item["name"] in vocab_dict:
            rep_vec[vocab_dict[item["name"]]] += 1
            # import pdb; pdb.set_trace()
    return rep_vec


def build_rep_vector_column(column, vocab_dict):
    rep_vec_col = []
    for row in column:
        rep_vec_col.append(build_rep_vec_row(row, vocab_dict))
    return rep_vec_col


def build_new_df(list_of_name_values_tuples):
    new_df = pd.DataFrame()
    for index, tup in enumerate(list_of_name_values_tuples):
        try:
            new_df.insert(index, *tup, allow_duplicates=False)
        except ValueError:
            continue
    return new_df



def data_manipulation(dataset):
    genres_col_vocab_tup = init_genres_vocab(dataset)
    cast_col_vocab_tup = init_cast_vocab(dataset)
    director_col_vocab_tup = init_crew_vocab(dataset, ['Director'])
    producer_col_vocab_tup = init_crew_vocab(dataset, ['Producer', 'Executive Producer'])
    keywords_col_vocab_tup = init_keyword_vocab(dataset)
    prod_company_col_vocab_tup = init_prod_vocab(dataset)
    prod_country_col_vocab_tup = init_production_countries_vocab(dataset)
    language_col_vocab_tup = init_language_vocab(dataset)

    col = dataset["release_date"]
    years_col, months_col = extract_release_year_month(col)

    list_of_name_values_tuples = [('id', dataset["id"])]
    list_of_name_values_tuples.append(('runtime', dataset["runtime"]))
    list_of_name_values_tuples.append(('release_year', years_col))
    list_of_name_values_tuples.append(('release_month', months_col))
    list_of_name_values_tuples.append(('genres', build_rep_vector_column(*genres_col_vocab_tup)))
    list_of_name_values_tuples.append(('cast', build_rep_vector_column(*cast_col_vocab_tup)))
    list_of_name_values_tuples.append(('director', build_rep_vector_column(*director_col_vocab_tup)))
    list_of_name_values_tuples.append(('producer', build_rep_vector_column(*producer_col_vocab_tup)))
    list_of_name_values_tuples.append(('keywords', build_rep_vector_column(*keywords_col_vocab_tup)))
    list_of_name_values_tuples.append(('production_company', build_rep_vector_column(*prod_company_col_vocab_tup)))
    list_of_name_values_tuples.append(('production_country', build_rep_vector_column(*prod_country_col_vocab_tup)))
    list_of_name_values_tuples.append(('spoken_language', build_rep_vector_column(*language_col_vocab_tup)))

    list_of_name_values_tuples.append(('budget', dataset["budget"]))
    list_of_name_values_tuples.append(('Revenue', dataset["revenue"]))


    logging.info('building new database')
    df = build_new_df(list_of_name_values_tuples)
    return df


def test(df):
    import pdb;
    pdb.set_trace()


def main():
    dataset = get_movies_db()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    import pdb;
    pdb.set_trace()
    df = dataset.head(50)
    df = data_manipulation(df)
    logistic_reg(df)

    pdb.set_trace()


if __name__ == "__main__":
    main()



