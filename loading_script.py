import numpy as np
import pandas as pd
import ast
from datetime import datetime as dt
TRAIN_CSV_PATH = './datasets/train.csv'
TEST_CSV_PATH = './datasets/test.csv'
CAST_NAME_REGEX = "(name\\':\ \\')([A-Z].+)(\\', \\'order)"
CAST_MAX_PRIO = 3
PROD_MAX_PRIO = 3

#filter thresholds
CAST_THRESH = 2
GENRE_THRESH = 5
CREW_THRESH = 1
PRODUCTION_THRESH = 10
KEYWORD_THRESH = 5
LANGUAGE_THRESH = 7
COUNTRY_THRESH = 5
DATE_THRESH = 2


def get_movies_db(path):
    pd.set_option("display.max_colwidth", 10000)
    dataset = pd.read_csv(path)
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

def geners_vocab_list(dataset):
    genres_vocab = []
    genre_col = dictioning_column(dataset["genres"])
    count = 0
    for movie in genre_col:
        genres_vocab.append([])
        for genre in movie:
            genres_vocab[count].append(genre["name"])
        count += 1
    return genres_vocab



def get_cast_list(cast,movie_id):
    cast_list = []
    cast_counter = 0
    for cast_member in cast[movie_id-1]:
        if cast_counter >= CAST_MAX_PRIO:
            break
        cast_list.append(cast_member["name"])
        cast_counter += 1
    return cast_list

def get_director_list(directors, movie_id):
    director_list = []
    for director in directors[movie_id-1]:
        if director["job"] == "Director":
            director_list.append(director["name"])
            break
    return director_list

def get_prod_comps_list(prod_comps, movie_id):
    prod_comp_list = []
    prod_comp_counter = 0
    for prod_comp in prod_comps[movie_id-1]:
        if prod_comp_counter >= PROD_MAX_PRIO:
            break
        prod_comp_list.append(prod_comp["name"])
        prod_comp_counter += 1
    return prod_comp_list

def get_languages_list(languages, movie_id):
    languages_list = []
    for language in languages[movie_id-1]:
        languages_list.append(language["iso_639_1"])
    return languages_list

def get_name_list(items, movie_id):
    items_list = []
    for item in items[movie_id - 1]:
        items_list.append(item["name"])
    return items_list

def get_release_year_list(release_dates, movie_id):
    date = dt.strptime(release_dates[movie_id-1], "%m/%d/%y")
    if date.year > dt.now().year:
        year = date.year - 100
    else:
        year = date.year
    return [year]

def get_release_month_list(release_dates, movie_id):
    date = dt.strptime(release_dates[movie_id-1], "%m/%d/%y")
    return [date.month]

def get_weekday_list(release_dates, movie_id):
    release_date = dt.strptime(release_dates[movie_id-1], "%m/%d/%y")
    if release_date.year > dt.now().year:
        year = release_date.year - 100
    else:
        year = release_date.year
    year  = str(year)
    month = str(release_date.month)
    day   = str(release_date.day)
    date_str = year + month + day
    s_datetime = dt.strptime(date_str, '%Y%m%d')
    weekday = s_datetime.weekday()
    return [weekday]

def get_is_collection_list(collections, movie_id):
    if not collections[movie_id-1]:
        return [0]
    else:
        return [1]

def get_num_words_list(titles, movie_id):
    title = titles[movie_id-1]
    return [len(title.split())]

def get_number_list(items,movie_id):
    item = items[movie_id-1]
    return [item]

def createDB(dataset):
    movieDB = {}
    ids = dataset["id"]
    genres      = dictioning_column(dataset["genres"])
    cast        = dictioning_column(dataset["cast"])
    directors   = dictioning_column(dataset["crew"])
    keywords    = dictioning_column(dataset["Keywords"])
    prod_comps  = dictioning_column(dataset["production_companies"])
    languages   = dictioning_column(dataset["spoken_languages"])
    countries   = dictioning_column(dataset["production_countries"])
    collections = dictioning_column(dataset["belongs_to_collection"])
    titles      = dataset["original_title"]
    date        = dataset["release_date"]
    budget      = dataset["budget"]
    runtime     = dataset["runtime"]
    popularity  = dataset["popularity"]
    revenue     = dataset["revenue"]
    for movie_id in ids:
        movieDB[movie_id] = {}
        movieDB[movie_id]["genres"]         = get_name_list(genres, movie_id)
        movieDB[movie_id]["cast"]           = get_cast_list(cast, movie_id)
        movieDB[movie_id]["director"]       = get_director_list(directors, movie_id)
        movieDB[movie_id]["keywords"]       = get_name_list(keywords, movie_id)
        movieDB[movie_id]["prod_companies"] = get_prod_comps_list(prod_comps, movie_id)
        movieDB[movie_id]["languages"]      = get_languages_list(languages, movie_id)
        movieDB[movie_id]["countries"]      = get_name_list(countries, movie_id)
        movieDB[movie_id]["is_collection"]  = get_is_collection_list(collections, movie_id)
        movieDB[movie_id]["num_words"]      = get_num_words_list(titles, movie_id)
        movieDB[movie_id]["release_year"]   = get_release_year_list(date, movie_id)
        movieDB[movie_id]["release_month"]  = get_release_month_list(date, movie_id)
        movieDB[movie_id]["weekday"]        = get_weekday_list(date, movie_id)
        movieDB[movie_id]["budget"]         = get_number_list(budget, movie_id)
        movieDB[movie_id]["runtime"]        = get_number_list(runtime, movie_id)
        movieDB[movie_id]["popularity"]     = get_number_list(popularity, movie_id)
        movieDB[movie_id]["revenue"]        = get_number_list(revenue, movie_id)
    return movieDB

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
    DB = createDB(dataset)
    #geners_vocab_list(dataset)
    print("test ended")

def main():
    train = get_movies_db(TRAIN_CSV_PATH)
    # testy = get_movies_db(TEST_CSV_PATH)
    test(train)
    print("end")

if __name__ == "__main__":
    main()

