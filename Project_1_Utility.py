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


def clean_calendar(df):
    # Drop missing values for those unavailable listings
    df = df.dropna()

    # Convert 'price' currency to numeric data
    df['price'] = df['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

    # Convert 'date' to datetime type
    df['date'] = pd.to_datetime(df['date'])

    return df


def time_series_analysis(df, city):
    # Clean
    df = clean_calendar(df)
    title1 = 'Price Trend in {} in 2016'.format(city)

    # Aggregate mean price by date
    tmp = df.groupby('date')[['price']].aggregate('mean')

    # Price by time plot
    plt.figure(figsize=(15, 8))

    sns.lineplot(x='date', y='price', data=tmp)
    sns.set_theme(font_scale=1.25)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title1, fontsize=18)

    plt.show()

    # Time series analysis
    analysis = tmp['price'].copy()
    decompose_result = seasonal_decompose(analysis, model="multiplicative")
    title2 = 'Seasonality Decomposition of Price in {}'.format(city)

    observe = decompose_result.observed
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid

    # Decomposition plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))

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
    # Create weekday column
    df = clean_calendar(df)
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()

    # Order x-axis categories
    df['weekday'] = pd.Categorical(df['weekday'],
                                   categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                               'Sunday'],
                                   ordered=True)

    # Group by weekday and month
    tmp = df.groupby(['month', 'weekday']).mean()[['price']].reset_index()

    # Visualization
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=tmp, x='weekday', y='price', hue='month', palette='deep')
    sns.set_theme(font_scale=1.25)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.title('Price by Weekday and Month', fontsize=18)
    plt.xlabel('Weekday')
    plt.ylabel('Price')

    plt.show()


def get_wordnet_pos(treebank_tag):

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
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:  # (compound score > -0.05) and (compound score < 0.05)
        return 'neutral/not applicable'


def word_freq_plot(text, sentiment, ax):
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
    ax.set_xlabel('Count')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16)


def sentiment_plot(df):
    # split positive and negative reviews
    pos = df[df['sentiment'] == 'positive']
    neg = df[df['sentiment'] == 'negative']

    # join text of each row into one big string
    pos_text = " ".join(text for text in pos['clean_text'])
    neg_text = " ".join(text for text in neg['clean_text'])

    # word frequency plot
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    fig1.suptitle("Word Frequency of Customer Reviews", fontsize=18)
    word_freq_plot(pos_text, 'Positive', ax1)
    word_freq_plot(neg_text, 'Negative', ax2)

    plt.show()

    # wordclouds
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
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
    # clean text
    df = df.dropna(subset=['comments'])
    df['clean_text'] = df['comments'].apply(clean_text)

    # analyze sentiment score using VADER model
    sid = SentimentIntensityAnalyzer()
    sentiments = df["clean_text"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df, sentiments.apply(pd.Series)], axis=1)

    # classify sentiment
    df['sentiment'] = df['compound'].apply(sentiment_lable)

    # sentiment plot
    tmp = df.sentiment.value_counts(normalize=True).reset_index()
    print(f'Sentiment distribution table:')
    print(tmp)

    plt.figure(figsize=(15, 9))
    sns.countplot(x="sentiment", data=df)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution', fontsize=18)
    plt.show()

    return df


def main():
    seattle_calendar = pd.read_csv('Seattle_data/calendar.csv')
    seattle_listings = pd.read_csv('Seattle_data/listings.csv')
    seattle_reviews = pd.read_csv('Seattle_data/reviews.csv')

    time_series_analysis(seattle_calendar)


if __name__ == '__main__':
    main()
