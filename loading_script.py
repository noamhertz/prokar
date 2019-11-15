import numpy as np
import pandas as pd
import ast
from datetime import datetime as dt
CSV_PATH = './datasets/train.csv'
CAST_NAME_REGEX = "(name\\':\ \\')([A-Z].+)(\\', \\'order)"
CAST_MAX_PRIO = 3

#filter thresholds
CAST_THRESH = 2
GENRE_THRESH = 5
CREW_THRESH = 1
PRODUCTION_THRESH = 10
KEYWORD_THRESH = 5
LANGUAGE_THRESH = 7
COUNTRY_THRESH = 5
DATE_THRESH = 2


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



def init_date_vocab(dataset, date_type):
    date_vocab = {}
    date_col = dataset["release_date"]
    movie_id = 0
    for release_date in date_col:
        movie_id += 1
        date = dt.strptime(release_date, "%m/%d/%y")
        if date_type == "year":
            if date.year > dt.now().year:
                year = date.year - 100
            else:
                year = date.year
            if year not in date_vocab:
                date_vocab[year] = [movie_id]
            else:
                date_vocab[year].append(movie_id)
        elif date_type == "month":
            if date.month not in date_vocab:
                date_vocab[date.month] = [movie_id]
            else:
                date_vocab[date.month].append(movie_id)
        elif date_type == "day":
            if date.day not in date_vocab:
                date_vocab[date.day] = [movie_id]
            else:
                date_vocab[date.day].append(movie_id)
    return filter_vocabs(date_vocab, DATE_THRESH)

def init_genres_vocab(dataset):
    genres_vocab = {}
    genre_col = dictioning_column(dataset["genres"])
    movie_id = 0
    for movie in genre_col:
        movie_id += 1
        for genre in movie:
            if genre["name"] not in genres_vocab:
                #Adding a new genre to vocab
                genres_vocab[genre["name"]] = [movie_id]
            else:
                genres_vocab[genre["name"]].append(movie_id)
    return filter_vocabs(genres_vocab, GENRE_THRESH)

def init_cast_vocab(dataset):
    cast_vocab = {}
    cast_col = dictioning_column(dataset["cast"])
    movie_id = 0
    for movie in cast_col:
        movie_id += 1
        cast_counter = 0
        for cast_member in movie:
            if cast_counter > CAST_MAX_PRIO:
                break
            if cast_member["name"] not in cast_vocab:
                # Adding a new cast member to vocab
                cast_vocab[cast_member["name"]] = [movie_id]
            else:
                cast_vocab[cast_member["name"]].append(movie_id)
            cast_counter += 1
    return filter_vocabs(cast_vocab, CAST_THRESH)

def init_keyword_vocab(dataset):
    keywords_vocab = {}
    keyword_col = dictioning_column(dataset["Keywords"])
    movie_id = 0
    for movie in keyword_col:
        movie_id += 1
        for keyword in movie:
            if keyword["name"] not in keywords_vocab:
                keywords_vocab[keyword["name"]] = [movie_id]
            else:
                keywords_vocab[keyword["name"]].append(movie_id)
    return filter_vocabs(keywords_vocab, KEYWORD_THRESH)

def init_crew_vocab(dataset,job):
    crew_vocab = {}
    movie_id = 0
    crew_col = dictioning_column(dataset["crew"])
    for movie in crew_col:
        movie_id += 1
        for crew_member in movie:
            if crew_member["job"] not in job:
                continue
            if crew_member["name"] not in crew_vocab:
                # Adding a new crew member to vocab
                crew_vocab[crew_member["name"]] = [movie_id]
            else:
                crew_vocab[crew_member["name"]].append(movie_id)
    return filter_vocabs(crew_vocab, CREW_THRESH)

def init_prod_vocab(dataset):
    prod_vocab = {}
    prod_comp_col = dictioning_column(dataset["production_companies"])
    movie_id = 0
    for movie in prod_comp_col:
        movie_id += 1
        for prod_comp in movie:
            if prod_comp["name"] not in prod_vocab:
                # Adding a new production company to vocab
                prod_vocab[prod_comp["name"]] = [movie_id]
            else:
                prod_vocab[prod_comp["name"]].append(movie_id)
    return filter_vocabs(prod_vocab, PRODUCTION_THRESH)

def init_language_vocab(dataset):
    lang_vocab = {}
    lang_col = dictioning_column(dataset["spoken_languages"])
    movie_id = 0
    for movie in lang_col:
        movie_id += 1
        for lang in movie:
            if lang["iso_639_1"] not in lang_vocab:
                # Adding a new production company to vocab
                lang_vocab[lang["iso_639_1"]] = [movie_id]
            else:
                lang_vocab[lang["iso_639_1"]].append(movie_id)
    return filter_vocabs(lang_vocab, LANGUAGE_THRESH)

def init_country_vocab(dataset):
    country_vocab = {}
    country_col = dictioning_column(dataset["production_countries"])
    movie_id = 0
    for movie in country_col:
        movie_id += 1
        for country in movie:
            if country["name"] not in country_vocab:
                # Adding a new production company to vocab
                country_vocab[country["name"]] = [movie_id]
            else:
                country_vocab[country["name"]].append(movie_id)
    return filter_vocabs(country_vocab, COUNTRY_THRESH)

def filter_vocabs(vocab, thresh):
    filtered_vocab = {}
    for item in vocab:
        if len(vocab[item]) >= thresh:
            filtered_vocab[item] = vocab[item]
    return filtered_vocab

def count_vocab(vocab):
    appearance_vocab = {}
    for item in vocab:
        appearance_vocab[item] = len(vocab[item])
    return sorted(appearance_vocab.items(), key=lambda x:x[1])


def test(dataset):
    # YEAR_VOCAB               = init_date_vocab(dataset, "year")
    # MONTH_VOCAB              = init_date_vocab(dataset, "month")
    # GENRES_VOCAB             = init_genres_vocab(dataset)
    # CAST_VOCAB               = init_cast_vocab(dataset)
    # DIRECTOR_VOCAB           = init_crew_vocab(dataset, ['Director'])
    # KEYWORDS_VOCAB           = init_keyword_vocab(dataset)
    # PRODUCTION_COMPANY_VOCAB = init_prod_vocab(dataset)
    # PRODUCER_VOCAB           = init_crew_vocab(dataset, ['Producer', 'Executive Producer'])
    # LANGUAGE_VOCAB           = init_language_vocab(dataset)
    # LANGUAGE_COUNT           = count_vocab(LANGUAGE_VOCAB)
    # COUNTRY_VOCAB            = init_country_vocab(dataset)
    # COUNTRY_COUNT            = count_vocab(COUNTRY_VOCAB)
    print("test ended")

def main():
    dataset = get_movies_db()
    test(dataset)
    print("end")

if __name__ == "__main__":
    main()

