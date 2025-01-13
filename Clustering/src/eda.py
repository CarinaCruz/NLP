# Databricks notebook source
import tqdm
from collections import Counter
from wordcloud import WordCloud
import seaborn as sea 
import numpy as np
import matplotlib.pyplot as plt


def plot_silhouette_scores(perplexity_silhouette):

    assert('scores' in perplexity_silhouette), 'Missing scores column!'

    keys = list(perplexity_silhouette['scores'].keys())
    values = list(perplexity_silhouette['scores'].values())
    plt.figure(figsize=(10, 6))
    plt.bar(keys, values, color='steelblue', alpha=0.7)
    plt.axhline(y=0.25, color='red', linestyle='--', linewidth=1, label='Acceptable score (0.25)')
    plt.title('Silhouette Scores by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Scores')
    plt.xticks(keys)
    plt.legend()
    plt.grid(axis='y', alpha=0.7)
    plt.show()

def most_common_word(text, max_words=10):
    counter = Counter(text.split(' '))
    counter = list(counter.keys())[0:max_words]
    counter= [c for c in counter if len(c) > 1]
    return counter   


def plot_word_cloud(df, text_column):
    cluster_text = df[text_column].values
    cluster_text = " ".join(cluster_text)
    cluster_text = [word for word in cluster_text.split(' ') if len(word) > 1]    
    cluster_text = " ".join(cluster_text)
    cluster_text = Counter(cluster_text.split(' '))
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(cluster_text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def generate_words_frequency(df, cluster, column, n=20):
    df['tokens'] = df[column].apply(lambda x: x.split())    
    counter = Counter([word for sublist in df['tokens'] for word in sublist])
    counter = Counter({key: value for key, value in counter.items() if len(key) > 3})
        
    top_words = dict(counter.most_common(n))

    plt.figure(figsize=(12, 6))
    plt.barh(list(top_words.keys()), list(top_words.values()), color='steelblue')
    plt.title(f'Top {n} words in cluster {cluster}')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.show()    


def plot_clusters_distribution(dataframe):
    grouped_df = dataframe.groupby(['Category', 'cluster'])['title'].count().to_frame().reset_index()
    grouped_df.columns = ['Category', 'Custer', 'News']
    plt.figure(figsize=(10, 6))
    sea.barplot(data=grouped_df, x='Custer', y='News', hue='Category', palette='Blues')
    plt.title('Categories Distributions per Cluster')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('News')
    plt.legend(title='Categories')
    plt.show()    