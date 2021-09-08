import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

pd.options.mode.chained_assignment = None  # default='warn'
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
from nltk import pos_tag
from nltk.corpus import wordnet
import folium
from folium import plugins


def clean_calendar(df):
    """
    The function clean_calendar is to clean the calendar dataset and make it ready for time series analysis.
    :param df: a pandas dataframe
    :return: a pandas dataframe
    """
    # Drop missing values for those unavailable listings
    df = df.dropna()

    # Convert 'price' currency to numeric data
    df['price'] = df['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

    # Convert 'date' to datetime type
    df['date'] = pd.to_datetime(df['date'])

    return df


def busiest_time(df, city):
    """
    The function busiest_time() is to explore the busiest time period of Airbnb homestays in Seattle.
    :param df: a pandas dataframe
    :param city: string, city name
    :return: a pandas dataframe
    """
    # clean
    df = clean_calendar(df)
    title1 = 'Booked Homestay Count in {} in 2016'.format('Seattle')

    # group by month
    df['month'] = df.date.dt.month

    groupby_month = df.groupby('month').count()[['price']].reset_index().rename(columns={'price': 'count'})

    # plotting
    plt.figure(figsize=(16, 8))

    sns.barplot(x='month', y='count', data=groupby_month)
    sns.set_theme(font_scale=1.25)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title(title1, fontsize=18)

    plt.show()

    # sorted table
    sorted_table = groupby_month.sort_values('count')

    return sorted_table


def time_series_analysis(df, city):
    """
    The function time_series_analysis() is to a time series analysis of room price in seattle, showing seasonality
    and trend plots.
    :param df: a pandas dataframe
    :param city: string, city name
    :return: null
    """
    # clean
    df = clean_calendar(df)
    title1 = 'Price Trend in {} in 2016'.format(city)

    # aggregate mean price by date
    tmp = df.groupby('date')[['price']].aggregate('mean')

    # price by time plot
    plt.figure(figsize=(16, 8))

    sns.lineplot(x='date', y='price', data=tmp)
    sns.set_theme(font_scale=1.25)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.title(title1, fontsize=18)

    plt.show()

    # time series analysis
    analysis = tmp['price'].copy()
    decompose_result = seasonal_decompose(analysis, model="multiplicative")
    title2 = 'Seasonality Decomposition of Price in {}'.format(city)

    observe = decompose_result.observed
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid

    # decomposition plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12))

    observe.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    residual.plot(ax=ax4)
    ax4.set_ylabel('Residual')

    fig.suptitle(title2, fontsize=18)
    fig.subplots_adjust(top=0.92, hspace=0.25)

    plt.show()


def weekday_decomposition(df):
    """
    The function weekday_decomposition is a further exploration of seasonality peaks, showing weekday price plots.
    :param df: a pandas dataframe
    :return: null
    """
    # create weekday column
    df = clean_calendar(df)
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()

    # Order x-axis categories
    df['weekday'] = pd.Categorical(df['weekday'],
                                   categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                               'Sunday'],
                                   ordered=True)

    # group by weekday and month
    tmp = df.groupby(['month', 'weekday']).mean()[['price']].reset_index()

    # visualization
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=tmp, x='weekday', y='price', hue='month', palette='deep')
    sns.set_theme(font_scale=1.25)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.title('Price by Weekday and Month', fontsize=18)
    plt.xlabel('Weekday', fontsize=16)
    plt.ylabel('Price', fontsize=16)

    plt.show()


def get_wordnet_pos(treebank_tag):
    """
    The function get_wordnet_pos() is to convert treebank word position to wordnet format.
    :param treebank_tag: string, word position in sentence
    :return: wordnet format position
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    """
    The function clean_text() is the preprocessing of text data fro sentiment analysis.
    :param text: string, paragraphs of reviews
    :return: cleaned text
    """
    # to lower text
    text = text.lower()

    # tokenize text into bag of words
    text = word_tokenize(text)

    # remove punctuation or numbers or uninterested tokens
    text = [word for word in text if word.isalpha()]

    # remove stop words
    stop_words = stopwords.words("english")
    text = [word for word in text if word not in stop_words]

    # pos tag text
    pos_tags = pos_tag(text)

    # Lexicon Normalization
    # performing stemming and Lemmatization
    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    text = [lem.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    text = [stem.stem(word) for word in text]

    # join together
    text = " ".join(text)

    return text


def sentiment_lable(score):
    """
    The function sentiment_lable() is to label the sentiment categories based on polarity score calculated by VADER model.
    :param score: float
    :return: string, sentiment label
    """
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:  # (compound score > -0.05) and (compound score < 0.05)
        return 'neutral/not applicable'


def word_freq_plot(text, sentiment, ax):
    """
    The function word_freq_plot() is to plot the most frequent words of positive and negative reviews.
    :param text: string, paragraphs for word frequency
    :param sentiment: string, 'positive' or 'negative'
    :param ax: the axis to plot
    :return: null
    """
    # word frequency
    words = text.split()
    word_freq = {}
    for w in words:
        if w in word_freq.keys():
            word_freq[w] += 1
        else:
            word_freq[w] = 1

    word_freq = pd.DataFrame(word_freq.items(), columns=['word', 'freq'])

    # plotting
    # top 30 frequent words
    tmp = word_freq.sort_values(by='freq', ascending=False).iloc[:30, :]
    title = '{} Reviews'.format(sentiment)

    sns.barplot(x='freq', y='word', data=tmp, ax=ax)
    sns.set_theme(font_scale=1.25)
    ax.set_xlabel('Count')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16)


def sentiment_word_plots(df):
    """
    The function word_freq_plot() is to plot the word frequency and word clouds.
    :param df:
    :return: null
    """
    # split positive and negative reviews
    pos = df[df['sentiment'] == 'positive']
    neg = df[df['sentiment'] == 'negative']

    # join text of each row into one big string
    pos_text = " ".join(text for text in pos['clean_text'])
    neg_text = " ".join(text for text in neg['clean_text'])

    # word frequency plot
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig1.suptitle("Word Frequency of Customer Reviews", fontsize=18)
    word_freq_plot(pos_text, 'Positive', ax1)
    word_freq_plot(neg_text, 'Negative', ax2)

    plt.show()

    # wordclouds
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 20))
    wc = WordCloud(background_color="white", max_words=1000, width=1500, height=900)

    ax1.imshow(wc.generate(pos_text), interpolation="bilinear")
    ax1.axis("off")
    ax1.set_title('Positive Reviews', fontsize=20)
    ax2.imshow(wc.generate(neg_text), interpolation="bilinear")
    ax2.axis("off")
    ax2.set_title('Negative Reviews', fontsize=20)
    fig2.suptitle('Wordclouds of Reviews', fontsize=24)
    fig2.subplots_adjust(top=0.93, hspace=0.15)

    plt.show()


def sentiment_analysis(df):
    """
    The function sentiment_analysis() is to analyze review sentiment using VADER model
    :param df: a pandas dataframe
    :return: a pandas dataframe with sentiment data added
    """
    # feature engineering
    # clean text
    df = df.dropna(subset=['comments'])
    df['clean_text'] = df['comments'].apply(clean_text)

    # analyze sentiment score using VADER model
    sid = SentimentIntensityAnalyzer()
    sentiments = df["clean_text"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df, sentiments.apply(pd.Series)], axis=1)

    # classify sentiment
    df['sentiment'] = df['compound'].apply(sentiment_lable)

    return df


def sentiment_distribution(df):
    """
    The function sentiment_distribution() is to plot an overview of sentiment distribution.
    :param df: a pandas dataframe
    :return: null
    """
    # sentiment plot
    tmp = df.sentiment.value_counts(normalize=True).reset_index()
    print(f'Sentiment distribution table:')
    print(tmp)

    plt.figure(figsize=(16, 9))
    sns.countplot(x="sentiment", data=df)
    sns.set_theme(font_scale=1.25)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution', fontsize=18)
    plt.show()


def listing_distribution_map(df, location):
    """
    The function listing_distribution_map() is to give an overview of listings distribution in Seattle area.
    :param df: a pandas dataframe
    :param location: a list of two elements, latitude and longitude of target area
    :return: a folium marker map object of listings scattered in Seattle, with room type legend and price popup
    """
    # plotting room type by price
    df['price'] = df['price'].replace({'\$': '', '%': '', ',': ''}, regex=True).astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    sns.set_theme(font_scale=1.25)
    sns.histplot(data=df, x='room_type', stat='probability', ax=ax1)
    sns.violinplot(x='room_type', y='price', data=df, ax=ax2)

    ax1.set_title('Room Type Frequency', fontsize=16)
    ax2.set_title('Price Distribution by Room Type', fontsize=16)

    plt.show()

    print(
        '\n============================================Visualization on Map==============================================\n')

    # visualize on map
    # Instantiate a feature group for the listings
    listing = folium.map.FeatureGroup()

    # Loop through the listings adn add to the map
    latitude = df.latitude
    longitude = df.longitude
    prices = df.price
    types = df.room_type

    for lat, lng, price, type in zip(latitude, longitude, prices, types):
        if type == 'Entire home/apt':
            listing.add_child(
                folium.CircleMarker(
                    location=[lat, lng],
                    popup=price,
                    radius=0.5,
                    color='red',
                )
            )
        elif type == 'Private room':
            listing.add_child(
                folium.CircleMarker(
                    location=[lat, lng],
                    popup=price,
                    radius=0.5,
                    color='green',
                )
            )
        else:
            listing.add_child(
                folium.CircleMarker(
                    location=[lat, lng],
                    popup=price,
                    radius=0.5,
                    color='yellow',
                )
            )

    # Add listings to map
    listing_map = folium.Map(location=location, zoom_start=12)
    listing_map.add_child(listing)

    return listing_map


def listing_count_map(df, location):
    """
    The function listing_count_map() is to give an overview of clusters of listings in Seattle area.
    :param df: a pandas dataframe
    :param location: a list of two elements, latitude and longitude of target area
    :return: a folium marker cluster object of listing count in each neighbourhood in Seattle
    """
    # plotting neighbourhood distribution
    plt.figure(figsize=(16, 8))
    sns.set(font_scale=1.25)
    sns.histplot(data=df, y='neighbourhood_group_cleansed', stat='probability')
    plt.ylabel('Neighbourhood', fontsize=16)
    plt.title('Neighbourhood Distribution', fontsize=18)

    plt.show()

    print(
        '\n============================================Visualization on Map==============================================\n')

    # visualize on map
    # instantiate the map of seattle
    neighbor_map = folium.Map(location=location, zoom_start=12)

    # instantiate a mark cluster object for the listings
    listing_count = plugins.MarkerCluster().add_to(neighbor_map)

    latitude = df.latitude
    longitude = df.longitude
    labels = df.neighbourhood

    # loop through the dataframe and add each data point to the mark cluster
    for lat, lng, label, in zip(latitude, longitude, labels):
        folium.Marker(
            location=[lat, lng],
            icon=None,
            popup=label,
        ).add_to(listing_count)

    # add neighbourhood count of listings to map
    neighbor_map.add_child(listing_count)

    return neighbor_map


def main():
    # import datasets
    calendar = pd.read_csv('Seattle_data/calendar.csv')
    reviews = pd.read_csv('Seattle_data/reviews.csv')

    # EDA for seattle
    # Time series analysis
    time_series_analysis(calendar, 'Seattle')
    weekday_decomposition(calendar)
    listings = pd.read_csv('Seattle_data/listings.csv')

    # Sentiment Analysis
    sentiment_df = sentiment_analysis(reviews)

    sentiment_distribution(sentiment_df)
    sentiment_word_plots(sentiment_df)

    # listing distribution visualization
    # seattle location
    seattle_location = [47.6062, -122.3321]

    listing_map = listing_distribution_map(listings, seattle_location)
    listing_map

    neighbourhood_map = listing_count_map(listings, seattle_location)
    neighbourhood_map


if __name__ == '__main__':
    main()
