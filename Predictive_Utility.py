import numpy as np
import pandas as pd


def initial_clean(df):
    # remove uninterested features
    cols = [col for col in df.columns if 'id' in col or 'url' in col or 'name' in col]

    # remove features with only one values
    cols += [col for col in df.columns if len(df[col].unique()) == 1]

    # manually filter out datetime/text variables or columns with homogenous information
    cols += ['summary', 'space', 'description', 'neighborhood_overview', 'notes',
             'transit', 'host_since', 'host_location', 'host_about', 'host_verifications',
             'street', 'neighbourhood', 'neighbourhood_cleansed', 'city', 'state', 'zipcode',
             'smart_location', 'amenities', 'calendar_updated', 'first_review', 'last_review']

    # drop unuseful features
    df = df.drop(cols, axis=1)

    return df


def clean_string_features(df):
    tmp = df[
        ['host_response_rate', 'host_acceptance_rate', 'price', 'weekly_price', 'monthly_price', 'security_deposit',
         'cleaning_fee', 'extra_people']]
    df_tmp = df.drop(
        ['host_response_rate', 'host_acceptance_rate', 'price', 'weekly_price', 'monthly_price', 'security_deposit',
         'cleaning_fee', 'extra_people'], axis=1)

    # remove dollar sign and percent sign
    tmp = tmp.replace({'\$': '', '%': '', ',': ''}, regex=True)

    # convert to numeric
    tmp = tmp.apply(pd.to_numeric)

    # convert percent to decimal
    tmp[['host_response_rate', 'host_acceptance_rate']] = tmp[['host_response_rate', 'host_acceptance_rate']].apply(
        lambda x: x / 100)

    # merge cleaned tmp to df
    df = pd.concat([df_tmp, tmp], axis=1)

    return df


def handle_missing_value(df):
    # drop features dominated with missing values (70%)
    df = df.drop(df.columns[df.isnull().sum() > 0.7 * df.shape[0]], axis=1)

    # drop weekly and monthly price
    # highly correlated with price, in case of multicollinearity
    # df = df.drop(['weekly_price', 'monthly_price'], axis=1)

    # split into categorical and numerical dataset
    # for numeric features, fill with mean; for categorical features, fill with mode
    cat = df[df.columns[df.dtypes == 'object']]
    num = df[df.columns[df.dtypes != 'object']]

    fill_mean = lambda col: col.fillna(col.mean())
    fill_mode = lambda col: col.fillna(col.mode()[0])

    cat = cat.apply(fill_mode, axis=0)
    num = num.apply(fill_mean, axis=0)
    df = pd.concat([cat, num], axis=1)

    return df




def clean_categorical_features(df):
    # check variability of categorical features
    # remove strongly imbalance features since they tell little information and remove features with too many levels
    # remove n(dominant) > 3200(84%) or n(levels) >10
    summary = df[df.columns[df.dtypes == object]].describe()
    cols = summary.columns[(summary.loc['freq'] > 3200) | (summary.loc['unique'] > 10)]
    df = df.drop(cols, axis=1)

    # create dummy variables
    df = pd.get_dummies(df, columns=)



def clean_listings(df):
    # clean dataset
    # initial clean unuseful features
    df = initial_clean(df)

    # clean string features
    # convert string numerical features into numeric datatype
    df = clean_string_features(df)

    # clean numerical features



def main():
    listings = pd.read_csv('Seattle_data/listings.csv')




if __name__ == '__main__':
    main()
