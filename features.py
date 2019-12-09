import numpy as np
import pandas as pd
import loading_script as ls
from datetime import datetime as dt
TRAIN_CSV_PATH = './datasets/train.csv'

def fill_holes(df):
    df.loc[df['id'] == 16, 'revenue'] = 192864  # Skinning
    df.loc[df['id'] == 90, 'budget'] = 30000000  # Sommersby
    df.loc[df['id'] == 118, 'budget'] = 60000000  # Wild Hogs
    df.loc[df['id'] == 149, 'budget'] = 18000000  # Beethoven
    df.loc[df['id'] == 313, 'revenue'] = 12000000  # The Cookout
    df.loc[df['id'] == 451, 'revenue'] = 12000000  # Chasing Liberty
    df.loc[df['id'] == 464, 'budget'] = 20000000  # Parenthood
    df.loc[df['id'] == 470, 'budget'] = 13000000  # The Karate Kid, Part II
    df.loc[df['id'] == 513, 'budget'] = 930000  # From Prada to Nada
    df.loc[df['id'] == 797, 'budget'] = 8000000  # Welcome to Dongmakgol
    df.loc[df['id'] == 819, 'budget'] = 90000000  # Alvin and the Chipmunks: The Road Chip
    df.loc[df['id'] == 850, 'budget'] = 90000000  # Modern Times
    df.loc[df['id'] == 1007, 'budget'] = 2  # Zyzzyx Road
    df.loc[df['id'] == 1112, 'budget'] = 7500000  # An Officer and a Gentleman
    df.loc[df['id'] == 1131, 'budget'] = 4300000  # Smokey and the Bandit
    df.loc[df['id'] == 1359, 'budget'] = 10000000  # Stir Crazy
    df.loc[df['id'] == 1542, 'budget'] = 1  # All at Once
    df.loc[df['id'] == 1570, 'budget'] = 15800000  # Crocodile Dundee II
    df.loc[df['id'] == 1571, 'budget'] = 4000000  # Lady and the Tramp
    df.loc[df['id'] == 1714, 'budget'] = 46000000  # The Recruit
    df.loc[df['id'] == 1721, 'budget'] = 17500000  # Cocoon
    df.loc[df['id'] == 1865, 'revenue'] = 25000000  # Scooby-Doo 2: Monsters Unleashed
    df.loc[df['id'] == 1885, 'budget'] = 12  # In the Cut
    df.loc[df['id'] == 2091, 'budget'] = 10  # Deadfall
    df.loc[df['id'] == 2268, 'budget'] = 17500000  # Madea Goes to Jail budget
    df.loc[df['id'] == 2491, 'budget'] = 6  # Never Talk to Strangers
    df.loc[df['id'] == 2602, 'budget'] = 31000000  # Mr. Holland's Opus
    df.loc[df['id'] == 2612, 'budget'] = 15000000  # Field of Dreams
    df.loc[df['id'] == 2696, 'budget'] = 10000000  # Nurse 3-D
    df.loc[df['id'] == 2801, 'budget'] = 10000000  # Fracture
    df.loc[df['id'] == 335, 'budget'] = 2
    df.loc[df['id'] == 348, 'budget'] = 12
    df.loc[df['id'] == 470, 'budget'] = 13000000
    df.loc[df['id'] == 513, 'budget'] = 1100000
    df.loc[df['id'] == 640, 'budget'] = 6
    df.loc[df['id'] == 696, 'budget'] = 1
    df.loc[df['id'] == 797, 'budget'] = 8000000
    df.loc[df['id'] == 850, 'budget'] = 1500000
    df.loc[df['id'] == 1199, 'budget'] = 5
    df.loc[df['id'] == 1282, 'budget'] = 9  # Death at a Funeral
    df.loc[df['id'] == 1347, 'budget'] = 1
    df.loc[df['id'] == 1755, 'budget'] = 2
    df.loc[df['id'] == 1801, 'budget'] = 5
    df.loc[df['id'] == 1918, 'budget'] = 592
    df.loc[df['id'] == 2033, 'budget'] = 4
    df.loc[df['id'] == 2118, 'budget'] = 344
    df.loc[df['id'] == 2252, 'budget'] = 130
    df.loc[df['id'] == 2256, 'budget'] = 1
    df.loc[df['id'] == 2696, 'budget'] = 10000000
    df.loc[df['id'] == 3033, 'budget'] = 250
    df.loc[df['id'] == 3051, 'budget'] = 50
    df.loc[df['id'] == 3084, 'budget'] = 337
    df.loc[df['id'] == 3224, 'budget'] = 4
    df.loc[df['id'] == 3594, 'budget'] = 25
    df.loc[df['id'] == 3619, 'budget'] = 500
    df.loc[df['id'] == 3831, 'budget'] = 3
    df.loc[df['id'] == 3935, 'budget'] = 500
    df.loc[df['id'] == 4049, 'budget'] = 995946
    df.loc[df['id'] == 4424, 'budget'] = 3
    df.loc[df['id'] == 4460, 'budget'] = 8
    df.loc[df['id'] == 4555, 'budget'] = 1200000
    df.loc[df['id'] == 4624, 'budget'] = 30
    df.loc[df['id'] == 4645, 'budget'] = 500
    df.loc[df['id'] == 4709, 'budget'] = 450
    df.loc[df['id'] == 4839, 'budget'] = 7
    df.loc[df['id'] == 3125, 'budget'] = 25
    df.loc[df['id'] == 3142, 'budget'] = 1
    df.loc[df['id'] == 3201, 'budget'] = 450
    df.loc[df['id'] == 3222, 'budget'] = 6
    df.loc[df['id'] == 3545, 'budget'] = 38
    df.loc[df['id'] == 3670, 'budget'] = 18
    df.loc[df['id'] == 3792, 'budget'] = 19
    df.loc[df['id'] == 3881, 'budget'] = 7
    df.loc[df['id'] == 3969, 'budget'] = 400
    df.loc[df['id'] == 4196, 'budget'] = 6
    df.loc[df['id'] == 4221, 'budget'] = 11
    df.loc[df['id'] == 4222, 'budget'] = 500
    df.loc[df['id'] == 4285, 'budget'] = 11
    df.loc[df['id'] == 4319, 'budget'] = 1
    df.loc[df['id'] == 4639, 'budget'] = 10
    df.loc[df['id'] == 4719, 'budget'] = 45
    df.loc[df['id'] == 4822, 'budget'] = 22
    df.loc[df['id'] == 4829, 'budget'] = 20
    df.loc[df['id'] == 4969, 'budget'] = 20
    df.loc[df['id'] == 5021, 'budget'] = 40
    df.loc[df['id'] == 5035, 'budget'] = 1
    df.loc[df['id'] == 5063, 'budget'] = 14
    df.loc[df['id'] == 5119, 'budget'] = 2
    df.loc[df['id'] == 5214, 'budget'] = 30
    df.loc[df['id'] == 5221, 'budget'] = 50
    df.loc[df['id'] == 4903, 'budget'] = 15
    df.loc[df['id'] == 4983, 'budget'] = 3
    df.loc[df['id'] == 5102, 'budget'] = 28
    df.loc[df['id'] == 5217, 'budget'] = 75
    df.loc[df['id'] == 5224, 'budget'] = 3
    df.loc[df['id'] == 5469, 'budget'] = 20
    df.loc[df['id'] == 5840, 'budget'] = 1
    df.loc[df['id'] == 5960, 'budget'] = 30
    df.loc[df['id'] == 6506, 'budget'] = 11
    df.loc[df['id'] == 6553, 'budget'] = 280
    df.loc[df['id'] == 6561, 'budget'] = 7
    df.loc[df['id'] == 6582, 'budget'] = 218
    df.loc[df['id'] == 6638, 'budget'] = 5
    df.loc[df['id'] == 6749, 'budget'] = 8
    df.loc[df['id'] == 6759, 'budget'] = 50
    df.loc[df['id'] == 6856, 'budget'] = 10
    df.loc[df['id'] == 6858, 'budget'] = 100
    df.loc[df['id'] == 6876, 'budget'] = 250
    df.loc[df['id'] == 6972, 'budget'] = 1
    df.loc[df['id'] == 7079, 'budget'] = 8000000
    df.loc[df['id'] == 7150, 'budget'] = 118
    df.loc[df['id'] == 6506, 'budget'] = 118
    df.loc[df['id'] == 7225, 'budget'] = 6
    df.loc[df['id'] == 7231, 'budget'] = 85
    df.loc[df['id'] == 5222, 'budget'] = 5
    df.loc[df['id'] == 5322, 'budget'] = 90
    df.loc[df['id'] == 5350, 'budget'] = 70
    df.loc[df['id'] == 5378, 'budget'] = 10
    df.loc[df['id'] == 5545, 'budget'] = 80
    df.loc[df['id'] == 5810, 'budget'] = 8
    df.loc[df['id'] == 5926, 'budget'] = 300
    df.loc[df['id'] == 5927, 'budget'] = 4
    df.loc[df['id'] == 5986, 'budget'] = 1
    df.loc[df['id'] == 6053, 'budget'] = 20
    df.loc[df['id'] == 6104, 'budget'] = 1
    df.loc[df['id'] == 6130, 'budget'] = 30
    df.loc[df['id'] == 6301, 'budget'] = 150
    df.loc[df['id'] == 6276, 'budget'] = 100
    df.loc[df['id'] == 6473, 'budget'] = 100
    df.loc[df['id'] == 6842, 'budget'] = 30

    return df

def dict2col(dict):
    return np.asarray(list(dict.values()))

def main():
    df = ls.get_movies_db(TRAIN_CSV_PATH)

    # dates
    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[(df['release_year'] <= 19) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[(df['release_year'] > 19) & (df['release_year'] < 100), "release_year"] += 1900
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter
    df = df.drop(columns=['release_date'])

    # collection
    df['is_collection'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), 'is_collection'] = 1
    df = df.drop(columns=['belongs_to_collection'])

    # homepage
    df['has_homepage'] = 0
    df.loc[pd.isnull(df['homepage']), 'has_homepage'] = 1
    df = df.drop(columns=['homepage'])

    # genres
    GENRES_VOCAB = ls.init_genres_vocab(df)
    df['num_genres'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['genres'])))
    df['genres_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['genres']), GENRES_VOCAB))
    df = df.drop(columns=['genres'])

    # keyword
    KEYWORDS_VOCAB = ls.init_keyword_vocab(df)
    df['num_keywords'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['Keywords'])))
    df['keywords_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['Keywords']), KEYWORDS_VOCAB))
    df = df.drop(columns=['Keywords'])

    # production companies
    PRODUCTION_COMPANY_VOCAB = ls.init_prod_vocab(df)
    df['num_prod_companies'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['production_companies'])))
    df['prod_companies_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['production_companies']), PRODUCTION_COMPANY_VOCAB))
    df = df.drop(columns=['production_companies'])

    # producers
    PRODUCER_VOCAB = ls.init_crew_vocab(df, ['Producer'])
    EXE_PRODUCER_VOCAB = ls.init_crew_vocab(df, ['Executive Producer'])
    num_producers = dict2col(ls.count_total_feature(ls.dictioning_column(df['crew']), 'job', 'Producer'))
    num_exe_producers = dict2col(ls.count_total_feature(ls.dictioning_column(df['crew']), 'job', 'Executive Producer'))
    df['num_producers'] = num_producers + num_exe_producers
    producers_populariy = dict2col(ls.count_popularity(ls.dictioning_column(df['crew']), PRODUCER_VOCAB))
    exe_producers_populariy = dict2col(ls.count_popularity(ls.dictioning_column(df['crew']), EXE_PRODUCER_VOCAB))
    df['producers_popularity'] = producers_populariy + exe_producers_populariy

    # crew & director
    DIRECTOR_VOCAB = ls.init_crew_vocab(df, ['Director'])
    df['num_crew'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['crew'])))
    df['num_director'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['crew']), 'job', 'Director'))
    df['director_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['crew']), DIRECTOR_VOCAB))

    # production countries
    COUNTRY_VOCAB = ls.init_country_vocab(df)
    df['num_prod_countries'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['production_countries'])))
    df['prod_countries_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['production_countries']), COUNTRY_VOCAB))
    df = df.drop(columns=['production_countries'])

    # language
    df['num_spoken_lang'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['spoken_languages'])))

    # cast
    CAST_VOCAB = ls.init_cast_vocab(df)
    df['num_cast'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['cast'])))
    df['num_females'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['cast']), 'gender', 1))
    df['is_lead_female'] = dict2col(ls.count_total_feature(ls.dictioning_column(df['cast']), 'gender', 1, 1))
    df['num_cast'] = df['num_cast'].apply(lambda x: 20 if x == 0 else x)
    df['cast_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['cast']), CAST_VOCAB))
    df['top5_popularity'] = dict2col(ls.count_popularity(ls.dictioning_column(df['cast']), CAST_VOCAB, top=5))
    df = df.drop(columns=['cast'])

    # budget
    df = fill_holes(df)
    df['inflation_budget'] = df['budget'] + df['budget'] * 1.8 / 100 * (2019 - df['release_year'])
    df['scaled_budget'] = np.log1p(df['budget'])
    df['budget_num_cast_ratio'] = df['budget'] / df['num_cast']
    df['budget_runtime_ratio'] = df['budget'] / df['runtime']



    print("end")

if __name__ == "__main__":
    main()



