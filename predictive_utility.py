import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def initial_clean(df):
    """
    The function initial_clean() is an initial cleaning of dataset by removing redundant columns
    :param df: an original pandas dataframe
    :return: initial cleaned pandas dataframe
    """
    # remove uninterested features
    cols = [col for col in df.columns if 'id' in col or 'url' in col or 'name' in col]

    # remove features with only one values
    cols += [col for col in df.columns if len(df[col].unique()) == 1]

    # manually filter out datetime/text variables, columns with homogenous information, or location data
    cols += ['summary', 'space', 'description', 'neighborhood_overview', 'notes',
             'transit', 'host_since', 'host_location', 'host_about', 'host_verifications',
             'street', 'neighbourhood', 'neighbourhood_cleansed', 'city', 'state', 'zipcode',
             'smart_location', 'amenities', 'calendar_updated', 'first_review', 'last_review',
             'latitude', 'longitude']

    # drop unuseful features
    df = df.drop(cols, axis=1)

    return df


def clean_string_features(df):
    """
    The function clean_string_features() is to clean string features by casting them to appropriate formats.
    :param df: an pandas dataframe
    :return: an pandas dataframe
    """
    # convert string numerical features into numeric datatype
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

    # merge features to df
    df = pd.concat([df_tmp, tmp], axis=1)

    return df


def clean_categorical_features(df):
    """
    The function clean_categorical_features() is to clean categorical features by removing feature with little
    variability and too many levels.
    :param df: an pandas dataframe
    :return: an pandas dataframe
    """
    # check variability of categorical features
    # remove strongly imbalanced features since they tell little information and remove features with too many levels
    # remove n(dominant) > 3200(84%) or n(levels) >10
    summary = df[df.columns[df.dtypes == object]].describe()
    cols = summary.columns[(summary.loc['freq'] > 3200) | (summary.loc['unique'] > 10)]
    df = df.drop(cols, axis=1)

    return df


def handle_missing_value(df):
    """
    The function handle_missing_value() is to handle missing values in the dataframe.
    :param df: an pandas dataframe
    :return: an pandas dataframe
    """
    # drop features dominated with missing values (70%)
    df = df.drop(df.columns[df.isnull().mean() > 0.7], axis=1)

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


def corr_heatmap(df):
    """
    The function corr_heatmap() is to show a correlation heatmap for all features in the dataframe.
    :param df: an pandas dataframe
    :return: null
    """
    # correlation heatmap
    plt.figure(figsize=(16, 16))
    sns.heatmap(df.corr())

    plt.show()


def clean_listings(df):
    """
    The function clean_listings() is to preprocess the dataset to make it ready for modeling.
    :param df: an pandas dataframe
    :return: an pandas dataframe
    """
    # clean dataset
    # initial clean unuseful features
    df = initial_clean(df)

    # feature engineering
    # convert string numerical features into numeric datatype
    df = clean_string_features(df)

    # clean categorical features
    df = clean_categorical_features(df)

    # handle missing values
    df = handle_missing_value(df)

    # encoding categorical features
    df = pd.get_dummies(df)

    return df


def train_fit_lr_model(X, y, test_size, rand_state):
    """
    The function train_fit_lr_model() is to train, fit, and evaluate the linear regression model for input dataset.
    :param X: a pandas dataframe, X matrix of all exploratory variables
    :param y: a pandas dataframe, response variable
    :param test_size: float between 0 and 1, the proportion of test set when splitting
    :param rand_state: int, controls random state for train_test_split
    :return:
    lm_model:  model object from sklearn
    X_train, X_test, y_train, y_test: split sets from train_test_split
    """
    # make a train/test split using 30% test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.drop(828, axis=0)
    y_test = y_test.drop(828, axis=0)

    # instantiate LinearRegression model
    lm_model = LinearRegression(normalize=True)

    # fit the model
    lm_model.fit(X_train, y_train)

    return lm_model, X_train, X_test, y_train, y_test


def initial_model(df, test_size=.30, rand_state=42):
    """
    The function initial_model() is to build initial linear regression model and show Actual vs Predicted plot.
    :param df: a pandas dataframe ready for modeling
    :param test_size: float between 0 and 1, default 0.30, the proportion of test set when splitting
    :param rand_state: int, default 42, controls random state for train_test_split
    :return:
    X: a pandas dataframe, exploratory variables
    y: a pandas dataframe, response variable
    """
    # split into explanatory and response variables
    X = df.drop('price', axis=1)
    y = df.loc[:, 'price']

    # train and fit linear regression model
    lm_model, X_train, X_test, y_train, y_test = train_fit_lr_model(X, y, test_size, rand_state)

    # predict response
    y_pred = lm_model.predict(X_test)

    # evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Actual vs Predicted graph
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    plt.title('Actual vs Predicted Price', fontsize=18)
    plt.xlabel('Predicted Price', fontsize=16)
    plt.ylabel('Actual Price', fontsize=16)
    plt.show()

    print("The initial model performance for testing set")
    print("--------------------------------------")
    print(f'MAE is {mae}')
    print(f'MSE is {mse}')
    print(f'R2 score is {r2}')

    return X, y


def find_optimal_lm_model(X, y, cutoffs, test_size=.30, rand_state=42, plot=True):
    """
    The function find_optimal_lm_model() is to try different number of dummy categorical vars and find the optimal model.
    :param X: a pandas dataframe, exploratory variables
    :param y: a pandas dataframe, response variable
    :param cutoffs: list of ints, cutoff for number of non-zero values in dummy categorical vars
    :param test_size: float between 0 and 1, default 0.30, the proportion of test set when splitting
    :param rand_state: int, default 42, controls random state for train_test_split
    :param plot: boolean, default True, True to plot result
    :return:
    best_r2_test: best r2 scores on the test data
    best_r2_train: best r2 scores on the train data
    lm_model: model object from sklearn
    X_train, X_test, y_train, y_test : split sets from train_test_split used for optimal model
    """

    r2_scores_test, r2_scores_train, num_features, results = [], [], [], dict()

    for cutoff in cutoffs:
        # reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_features.append(reduce_X.shape[1])

        # split the data into train and test
        # train and fit linear regression model
        lm_model, X_train, X_test, y_train, y_test = train_fit_lr_model(reduce_X, y, test_size, rand_state)

        # obtain predicted response
        y_test_pred = lm_model.predict(X_test)
        y_train_pred = lm_model.predict(X_train)

        # append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_pred))
        r2_scores_train.append(r2_score(y_train, y_train_pred))
        results[str(cutoff)] = r2_score(y_test, y_test_pred)

    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(num_features, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_features, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features', fontsize=16)
        plt.ylabel('Rsquared', fontsize=16)
        plt.title('Rsquared by Number of Features', fontsize=18)
        plt.legend(loc=2, fontsize=16)
        plt.show()

    best_cutoff = max(results, key=results.get)

    # reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    # num_features.append(reduce_X.shape[1])

    # split the data and fit the model
    lm_model, X_train, X_test, y_train, y_test = train_fit_lr_model(reduce_X, y, test_size, rand_state)

    # obtain predicted response
    y_test_pred = lm_model.predict(X_test)
    y_train_pred = lm_model.predict(X_train)

    # best score
    best_r2_test = r2_score(y_test, y_test_pred)
    best_r2_train = r2_score(y_train, y_train_pred)

    print("The model performance for optimal linear regression model")
    print("------------------------------------------------------------")
    print(f'Number of Features: {X_train.shape[1]}')
    print(f'R2 score for test set: {best_r2_test}')
    print(f'R2 score for training set: {best_r2_train}')

    return best_r2_test, best_r2_train, lm_model, X_train, X_test, y_train, y_test


def coef_weights(lm_model, X_train):
    """
    The function coef_weights() is to explore the importance of each feature on response variable.
    :param lm_model: linear regression model object from sklearn
    :param X_train: training X matrix, for extracting column names
    :return: a pandas dataframe of coefficient estimate and abs(estimate) for each exploratory variables
    """

    coefs_df = pd.DataFrame()
    coefs_df['predictors'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)

    # plotting
    tmp = coefs_df.sort_values('coefs').reset_index(drop=True).drop(index=0)
    plt.figure(figsize=(15, 12))
    sns.set_theme()
    sns.barplot(x='coefs', y='predictors', data=tmp)
    plt.title('Model Coefficients', fontsize=18)
    plt.ylabel('Predictors', fontsize=16)
    plt.xlabel('Coefficients', fontsize=16)

    plt.show()

    return coefs_df


def main():
    listings = pd.read_csv('Seattle_data/listings.csv')

    # Predictive modeling for Airbnb 'price' feature
    # Clean dataset
    df = clean_listings(listings)

    # Modeling
    # Initial model
    X, y = initial_model(df, test_size=.30, rand_state=42)

    # %%
    # Find the optimal model by changing training features
    cutoffs = [3000, 2000, 1000, 500, 100, 50, 30, 25]
    best_r2_test, best_r2_train, lm_model, X_train, X_test, y_train, y_test = find_optimal_lm_model(X, y, cutoffs)

    # %%
    # feature importance
    coef_df = coef_weights(lm_model, X_train)
    coef_df


if __name__ == '__main__':
    main()
