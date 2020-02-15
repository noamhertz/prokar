import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def main():
    dataframe = pd.read_csv('datasets/filtered_dataset.csv')
    '''
    ## realese year
    sns.countplot(dataframe['release_year'].sort_values())
    plt.title("Movie Release count by Year",fontsize=20)
    loc, labels = plt.xticks()
    plt.xticks(fontsize=10, rotation=90)
    plt.show()


    ## release Q
    ax = sns.countplot(dataframe['release_quarter'].sort_values())
    plt.title("Movie Release Quarter",fontsize=20)
    loc, _ = plt.xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    ax.set_xticklabels(labels)
    plt.xticks(fontsize=12, rotation=90)
    plt.show()

    ## release DOW
    ax = sns.countplot(dataframe['release_dayofweek'].sort_values())
    plt.title("Movie Release count by DayOfWeek",fontsize=20)
    loc, _ = plt.xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    ax.set_xticklabels(labels)
    plt.xticks(fontsize=11, rotation=0)
    plt.show()
    '''
    '''
    ## Lead Actor Gender
    ax = sns.countplot(dataframe['is_lead_female'].sort_values(), palette=['C0', 'C1'])
    plt.title("Leading Role Gender",fontsize=20)
    loc, _ = plt.xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['Male', 'Female']
    ax.set_xticklabels(labels)
    plt.xlabel('Gender')
    plt.xticks(fontsize=12, rotation=0)
    plt.show()
    '''
    ## revenue vs is lead female
    '''
    group_by_gender = dataframe.groupby('is_lead_female').count()
    count_by_gender = group_by_gender['id']
    sum_by_gender = dataframe.groupby('is_lead_female').sum()
    revenue_by_gender = sum_by_gender['revenue']
    ylist = np.log10(revenue_by_gender /count_by_gender)
    bar = plt.bar(['Male', 'Female'], ylist)
    bar[1].set_color('C1')
    plt.ylim(min(ylist)-0.05, max(ylist)+0.05)
    plt.title("Average Movie Budget by Quarter", fontsize=20)
    plt.xlabel('Gender')
    plt.ylabel('Log Average Movie Budget')
    plt.show()
    '''

    ## Budget Scatter
    '''
    ax = sns.scatterplot(np.log10(dataframe['budget'].sort_values()), np.log10(dataframe['revenue']))
    plt.title("Log of Revenue vs Log Budget Scatter",fontsize=20)
    loc, labels = plt.xticks()
    plt.xlabel('Log of Budget')
    plt.ylabel('Log of Revenue')
    plt.xticks(fontsize=12, rotation=90)
    plt.show()
    '''

## Budget Runtime Scatter
    '''
    ax = sns.scatterplot(np.log10(dataframe['budget'].sort_values()), dataframe['runtime'])
    plt.title("Runtime vs Log Budget Scatter",fontsize=20)
    loc, labels = plt.xticks()
    ax.set(xlabel='Log of Budget', ylabel='Runtime, Minutes')
    plt.xticks(fontsize=12, rotation=90)
    plt.show()
    '''


## budget vs release Q
    '''
    group_by_q = dataframe.groupby('release_quarter').count()
    count_by_q = group_by_q['id']
    sum_by_q = dataframe.groupby('release_quarter').sum()
    budget_by_q = sum_by_q['budget']
    ylist = np.log10(budget_by_q / count_by_q)
    plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], ylist)
    plt.ylim(min(ylist)-0.1, max(ylist)+0.1)
    plt.title("Average Movie Budget by Quarter", fontsize=20)
    plt.xlabel('Quarter')
    plt.ylabel('Log Average Movie Budget')
    plt.show()
    '''

## spoken language
    '''
    group_by_lang = dataframe.groupby('language').count()['id']
    english_films = group_by_lang[0]
    other_lang_films = sum(group_by_lang[1:])
    plt.bar(['English Language Films', 'Other Language Films'], [english_films, other_lang_films])
    plt.title("Spoken Language Films", fontsize=20)
    plt.xlabel('Spoken Language')
    plt.ylabel('Count')
    plt.show()
    '''

## producation country
    '''
    us_prod = dataframe.loc[dataframe['prod_countries_popularity'] == 1790]
    other_prod = dataframe.loc[dataframe['prod_countries_popularity'] != 1790]
    us_size = us_prod.shape[0]
    other_size = other_prod.shape[0]
    plt.bar(['US Films', 'Other Countries Films'], [us_size, other_size])
    plt.title("Production  Country Films", fontsize=20)
    plt.xlabel('Production Country')
    plt.ylabel('Count')
    plt.show()
    '''
    '''
    ## high budget adjacency by qurater
    group_by_q = dataframe.groupby('release_quarter').count()
    count_by_q = group_by_q['id']
    sum_by_q = dataframe.groupby('release_quarter').sum()
    adj_by_q = sum_by_q['high_budget_adjacency']
    ylist = adj_by_q/ count_by_q
    plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], ylist)
    plt.ylim(min(ylist) - 0.1, max(ylist) + 0.1)
    plt.title("Average Movie High Budget Adjacency by Quarter", fontsize=20)
    plt.xlabel('Quarter')
    plt.ylabel('Average High budget Adjacency')
    plt.show()
    '''

    '''
    ## belongs to a collection
    ax = sns.countplot(dataframe['is_collection'].sort_values())
    plt.title("Is Movie Part of a Collection", fontsize=20)
    loc, _ = plt.xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['No', 'Yes']
    plt.xlabel('Is Part of a Collection')
    ax.set_xticklabels(labels)
    plt.xticks(fontsize=12, rotation=0)
    plt.show()
    '''

    ## top5 popularity vs budget
    ax = sns.scatterplot(np.log10(dataframe['budget'].sort_values()), (dataframe['top5_popularity']))
    plt.title("Cast Top5 popularity vs Log Budget Scatter", fontsize=20)
    loc, labels = plt.xticks()
    plt.xlabel('Log of Budget')
    plt.ylabel('Cast Top5 popularity')
    plt.xticks(fontsize=12, rotation=90)
    plt.show()

if __name__ == "__main__":
    main()