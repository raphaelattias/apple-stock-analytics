# Import all the required libraries
import os
import pandas as pd
import numpy as np
import pageviewapi 
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import iplot

# Import functions from python files
from util.sentiment_analysis import *
from util.dataloader import *


# For having tqdm progress bar when we use apply function
tqdm.pandas()


# ----------------------------------------------------------------- #


def pandas_process(input):
    """
        This function is used to transform all the qids in the 
        dataframe to a string format

    Args:
        input [any other format]: Just an object we want to transform 
        into a string (in our case it will be the QID)

    Returns:
        [string]: The input ttranformed into a string.
    """

    # Transfrom into a string
    output = str(input)

    # Return the output
    return output


# ----------------------------------------------------------------- #


def get_speakers_ids(quotes):
    """
        This function is here to get all the ID of the different speakers in 
        the wiki data set. The idea is to have list of all the person that
        have spoken in Quotebank da  ta set and keep only the wikipedia 
        information we want. 

    Args:
        quotes [pd.Dataframe]: Data frame that contain all teh quotes

    Returns:
        [pd.Dataframe]: 
    """

    # Copy the data frame
    wiki_quotes = quotes.copy()

    # Apply the string filter and drop all the duplicates
    wiki_quotes['qids'] = wiki_quotes['qids'].apply(pandas_process).astype('|S')
    wiki_quotes = wiki_quotes.drop_duplicates(['qids'])

    # Get the indices to drop
    indices_to_keep = np.array(wiki_quotes.index)
    # indices_to_drop = set(all_indices) - set(indices_to_keep)
    wiki_quotes = quotes.iloc[indices_to_keep]

    # Get the list of ID's
    wiki_quotes = wiki_quotes[['speaker', 'qids']]
    wiki_quotes = wiki_quotes.dropna()
    wiki_quotes = wiki_quotes.drop(wiki_quotes[wiki_quotes.speaker == 'None'].index).reset_index(drop = True)

    # Return the result
    return wiki_quotes


# ----------------------------------------------------------------- #


def get_wiki_labels():
    """
    The idea of this function is to concatenate all the wiki files
    for the speaker attributes in one big file. The goal is not to keep
    a big file and keep surching for information inside, we will do a
    filtering with all the qids of all the speakers we have.

    Returns:
        [DataFrame]: Returned all files in one dataframe.
    """
    
    # Initialization of the files variable
    files = []

    # Initailize the category
    category = 'wiki speakers attributes'

    # Initialize the path
    path = 'data/wiki_speaker_attributes/'
    path = os.path.join(os.getcwd(), path)

    # Get the dictionniary conatining all the adresses
    wiki_dict = get_drive_dictionnary()[category]

    # Loop over all the wiki files containing speaker attributes
    for key in wiki_dict:
        
        # Get the path of the current file
        current_path = os.path.join(path, key)

        # If the file is not dowloaded in the directory, we download it.
        if not os.path.isfile(current_path):
            download(current_path, category)

        # Append the list files containing all the previous files
        files.append(pd.read_parquet(current_path))

    # Concatenating all the files in a single data frame
    df_wiki = pd.concat(files)

    # Return the result
    return df_wiki


# ----------------------------------------------------------------- #


def remove_duplicates(df, column_name):
    """
    The idea of this function is to remove all the duplicated lines 
    we could have in our dataframe.

    Args:
        df [Dataframe]: Input dataframe.
        column_name ([type]): Column where we want to remove the 
        duplicates.

    Returns:
        [DataFrame]: Cleaned dataframe (without duplicates)
    """

    # Intializa ou cleaned dataframe
    df_cleaned = df.copy()

    # Apply a string filter and drop the duplicates
    df_cleaned[column_name] = df_cleaned[column_name].apply(pandas_process).astype('|S')
    df_cleaned = df_cleaned.drop_duplicates([column_name])

    # Return the cleaned dataframe
    return df_cleaned


# ----------------------------------------------------------------- #


def get_page_views_per_year(page, year):
    """
    The idea of this function is to get the number of wiki pageviews
    on one specific year.

    Args:
        page [string]: The name of the Wikipedia's page
        year [integer]: Year when we want the number of pageviews

    Returns:
        [integer]: number of pageviews.
    """

    # Initialization of the number of pageviews
    nb_page_views = None

    # Check if the name of the page is well defined
    if str(page) != 'None':

        # Time line
        begin = str(year) + '0101'
        end = str(year) + '1231'

        # Try if the page exist or not
        try:
            # Get the data frame from wikipedia API
            df_wiki_api = pd.DataFrame(pageviewapi.per_article('en.wikipedia', page, begin, end,
                            access='all-access', agent='all-agents', granularity='monthly')['items'])
        
            # Get the number of page views
            nb_page_views = df_wiki_api.views.sum()

        except:
             pass

    # Return the results
    return nb_page_views


# ----------------------------------------------------------------- #


def get_total_page_views(page):
    """
    Get the total number of pageviews between 2015 and 2020 of this
    specific page.

    Args:
        page (string): The name of the page

    Returns:
        [integer]: Total number of pageviews
    """

    # Get an array with every pageviews number between 2015 and 2020
    page_views = np.array([get_page_views_per_year(page, year) for year in range(wiki_page_year(page), 2021)])

    # Sum all the values
    total_nb = np.sum(page_views)

    # Return the final result
    return total_nb


# ----------------------------------------------------------------- #


def exist_wiki_page_for_year(page, year):
    """
    This function returns if the page exists for a particular year.

    Args:
        page [string]: the name of the wiki page
        year [integer]: the year when we want to check the existence 
        of the page.

    Returns:
        [boolean]: returns if the page exists.
    """

    # Check if the function works
    try :
        get_page_views_per_year(page, year)
        return True

    except:
        return False


# ----------------------------------------------------------------- #


def wiki_page_year(page):
    """
    This function return the first year when the page exists between
    2015 and 2020. The idea is because some speakers has there wiki 
    page that was created in 2018 for example.

    Args:
        page [string]: the name of the wiki page

    Returns:
        [integer]: return the first year between 2015 and 2020.
    """

    # Check wether this speaker has a wiki page
    for year in range(2015, 2021):
        if exist_wiki_page_for_year(page, year):
            # Return the first year when it exists
            return year

    # Return 2021 if it has not one between 2015 and 2020 to be outrange
    return 2021


# ----------------------------------------------------------------- #


def exist_wiki_page(page):
    """
    The idea of this function is to know if the page exists between
    2015 and 2020.

    Args:
        page (string): The name of the page

    Returns:
        [Boolean]: True/False if the page exists or not.
    """

    # Check wether this speaker has a wiki page
    for year in range(2015, 2021):
        if exist_wiki_page_for_year(page, year):
            return True

    # Return flase if it has not one
    return False


# ----------------------------------------------------------------- #


def wiki_label_speaker(ids, wiki_data):
    """
    This function is used in the apply function on the dataframe
    containing all the speakers list, and it returns for each row
    the label of the speaker (found in wiki_data) having the 
    maximum number of pageviews.

    Args:
        ids (list): List of string containing the qids of this speaker
        wiki_data (Dataframe): It contains all the ids wiki page with
        there exact label.

    Returns:
        string: Return the exact label.
    """

    # Initialization
    label = None
    prev_nb_views = 0

    # Go through all the qids containing in the list ids
    for id in ids:

        try:
            # Get the label of the current id and it total pageviews
            speaker = wiki_data.label[wiki_data.id == id].reset_index(drop = True).values[0]
            total_page_views = get_total_page_views(speaker)

            # Check if it exists and the pageviews is the larger than the previous label
            if exist_wiki_page(speaker) and total_page_views > prev_nb_views:
                # Update the label and the max number of pageviews
                label = speaker
                prev_nb_views = total_page_views

        except:
            pass
        
    # Return the kept label
    return label


# ----------------------------------------------------------------- #


def save_speakers_id(speakers_id, save, id_nb):
    """
    This function is used for the clustering of the run. Since the run
    is long, we do not want to do the run again because the computer
    crashes at 90 %. So we save small steps, to be sure to keep some
    of the run if one of the computer crashes.

    Args:
        speakers_id (Dataframe): Dataframe to save
        save (Boolean): Know if we want to save it or not
        id_nb (integer): ID of the save, so that we can reconstruct the 
        whole dataframe.
    """

    # Check if we want to save teh file
    if save:

        # Set up the name of the file
        name = 'speakers_labels_' + str(id_nb)
        save_path = os.path.join('data/wiki_speaker_attributes', name+ '.pkl')

        # Notify the user that the file already exists and will be deleted
        if os.path.isfile(save_path):
            print(f"WARNING: the file {save_path} already exists and will be deleted at the end.")

        # Save the file
        speakers_id.to_pickle(save_path)


# ----------------------------------------------------------------- #


def find_labels(speakers_id, wiki_data):
    """
    This function is used to add to the dataframe with all the 
    speakers a new column label that corresponds to the exact label
    of each speaker.

    Args:
        speakers_id (Dataframe): It contains all the speaker's id 
        wiki_data (Dataframe): It contains all the id with the 
        corresponding wiki label page.

    Returns:
        Dataframe: Returns the new dataframe with the added column label
    """

    # Copy the dataframe, to be sure that we can rerun this code multiple times
    speakers_id_new = speakers_id.copy()

    # Get the new column label
    speakers_label = speakers_id.qids.progress_apply(lambda ids: wiki_label_speaker(ids, wiki_data))

    # Add the new column to the dataframe
    speakers_id_new['label'] = speakers_label

    # Return the result
    return speakers_id_new


# ----------------------------------------------------------------- #


def add_labels(speakers_id, wiki_data, save = False, cluster = None):
    """
    This function was used to construct the whole dataframe with all 
    the speakers and there exact label for there wikipage. The idea
    was to run this function on 5 different computers by setting the
    number of computer cluster = [computer_nb]. We save for each 
    computer, the dataframe in four small datframes to be sure that if
    the computer crashes during the run, we lose not everything. So, 
    at the end, we have 5x4 = 20 files to concatenates with the 
    following function 'concat_save_all_speakers_labels' and save in
    one single file.

    Args:
        speakers_id (Dataframe): whole dataframe that contains all the
        speakers ids.
        wiki_data (Dataframe): it contains all the wiki id with the 
        corresponding exact label.
        save (bool, optional): To save or not the dataframes. Defaults to False.
        cluster (integer, optional): To precise what is the number of 
        the computer. Defaults to None.

    Returns:
        [dataframe]: Returns the new datafralme with the exact label for
        all speakers.
    """

    # If we are not using the clustering method by using multiple computer.
    # Mainly to test the good process of the function.
    if cluster == None:

        # Copy the dataframe
        speakers_id_new_df = speakers_id.copy()

        # Initilization of some indices for batch
        nb_elem = speakers_id_new_df.shape[0]
        batch_size = nb_elem / 20

        # Get the first step of the loop
        speakers_id_batch = speakers_id.copy().iloc[np.arange(0, batch_size).astype(int)]
        speakers_id_batch = find_labels(speakers_id, wiki_data)

        # Save (or not) the dataframe
        save_speakers_id(speakers_id_batch, save, 1)

        # Loop over all th 19 steps remaining
        for batch in range(1, 20):

            # Update the indices for batches
            batch_begin = (batch * batch_size) + 1
            batch_end = min(nb_elem, batch_begin + batch_size)

            # Set the new batch fo the dataframe where we want to work
            speakers_id_batch = speakers_id.copy().iloc[np.arange(batch_begin, batch_end).astype(int)]

            # Find the labels for this new dataframe
            speakers_id_batch = find_labels(speakers_id_batch, wiki_data)

            # Save (or not) the current batch
            save_speakers_id(speakers_id_batch, save, batch + 1)

    # If we are using clustering method
    else:

        # Copy the dataframe
        speakers_id_new_df = speakers_id.copy()

        # Initilization of some indices for batch   
        nb_elem = speakers_id_new_df.shape[0]
        sub_set_size = nb_elem / 4

        # Set begin and end of the batch fo the dataframe 
        begin = (cluster - 1) * sub_set_size
        if cluster == 4:
            end = nb_elem
        else:
            end = (cluster * sub_set_size) - 1

        # Batch indices
        sub_set_nb_elem = end - begin + 1
        batch_size = sub_set_nb_elem / 5

        # Loop on the 5 mini-files for this cluster
        for batch in range(0, 5):

            # Update the batch range
            batch_begin = begin + (batch_size * batch)
            batch_end = min(end, batch_begin + batch_size - 1)

            # Get the batch from the dataframe
            speakers_id_batch = speakers_id.copy().iloc[np.arange(batch_begin, min(end, batch_end + 1)).astype(int)]

            # Add the new column to the batched dataframe
            speakers_id_batch = find_labels(speakers_id_batch, wiki_data)

            # Set the ID number of the dataframe depending of batch nb and the cluster
            id_nb = ((cluster - 1) * 5) + 1 + batch

            # Save (or not) the current file
            save_speakers_id(speakers_id_batch, save, id_nb)
    
    # Get back all the files in ones dataframe and save it
    speakers_id_new_df = concat_save_all_speakers_labels()

    # Return the dataframe
    return speakers_id_new_df


# ----------------------------------------------------------------- #


def concat_save_all_speakers_labels():
    """
    This function concatenates all the 20 files created by the 
    previous function in one single dataframe and we save it, such as
    we do not need anymore to rerun the previous function with all the
    data.

    Returns:
        [Dataframe]: Concatenated dataframe.
    """

    # Get the path and the file name pattern for the 20 files
    folder_path = 'data/wiki_speaker_attributes/'
    file_name = 'speakers_labels_'
    
    # Get the first file in a dataframe
    speakers_labels = pd.read_pickle(os.path.join(folder_path + file_name + '1.pkl'))

    # Loop along the 19 remaining files
    for idx_file in range(1, 21):

        # Get the path of the i-th file
        path = os.path.join(folder_path + file_name + str(idx_file) + '.pkl')

        # Get the current dataframe
        current_speakers_labels = pd.read_pickle(path)

        # Concatenate the dataframe with the others in a single one
        speakers_labels = pd.concat([speakers_labels, current_speakers_labels])
    
    # Drop duplicates along the speaker, it can appear due to the batch 
    # cut that is not perfect.
    speakers_labels = speakers_labels.drop_duplicates(subset=['speaker']).reset_index(drop = True)

    # Save the Dataframe in a pickle file
    save_path = os.path.join(folder_path + 'speakers_labels.pkl')
    speakers_labels.to_pickle(save_path)

    # Return the one-file dataframe
    return speakers_labels


# ----------------------------------------------------------------- #


def scoring(quote_row):
    """
    This function is used to get the number of pageviews for the 
    speaker that produces the quote.

    Args:
        quote_row (line of dataframe): The line of the dataframe that
        contains the speaker, the year when the speaker was cited and
        the number of pageviews per year of the speaker for ther years
        between 2015 and 2020.

    Returns:
        [integer]: the number of pageviews of the speaker at the year 
        when he was cited.
    """

    try:
        # Update the score
        score = quote_row[str(quote_row['year'])]
    except:
        # Set None if there is no socre
        score = None
    
    # Return the result
    return score


# ----------------------------------------------------------------- #


def get_pageview_quotes(quotes, speakers_pageviews):
    """
    This function add a new column pageviews to the dataframe quotes
    that contains the number of pageviews of the speaker at the year 
    when he was cited for each quotes. 

    Args:
        quotes (dataframe): it contains all the quotes
        speakers_pageviews ([type]): contains all the speaker with
        their exact label and the number of pageviews for every years
        between 2015 and 2020.

    Returns:
        [Dataframe]: Dataframe quotes with the new column pageviews
    """

    # Copy the dataframe
    quotes_pageviews = quotes.copy()

    # Add a column year for practice and keep the only ones after 2015
    quotes_pageviews['year'] = quotes_pageviews.date.apply(lambda date: date.year)
    quotes_pageviews = quotes_pageviews[quotes_pageviews.year >= 2015]

    # Add the columns label and all the pageviews of speaker to every
    # quotes in the dataframe bu using join function.
    quotes_pageviews = quotes_pageviews.set_index('speaker').join(speakers_pageviews.set_index('speaker'), 
                                        lsuffix="_left", 
                                        rsuffix="_right").reset_index()
    
    # Create the new column pageviews
    quotes_pageviews['pageviews'] = quotes_pageviews.apply(scoring, axis = 1)

    # Drop the useless columns
    quotes_pageviews = quotes_pageviews.drop(['2015', '2016', '2017', '2018', '2019', '2020', 'year'], axis = 1).reset_index(drop = True)

    # Return the new dataframe
    return quotes_pageviews


# ----------------------------------------------------------------- #


def get_score_quotes(quotes, speakers_pageviews):
    """
    The idea now, is to use the previous function to get the number
    of pageviews for every quotes. Then we do a min/max normalization
    of this column to create a new one corresponding to the score.

    Args:
        quotes (dataframe): dataframe containing all quotes
        speakers_pageviews (dataframe): contains all the speakers labels
        and the pageviews.

    Returns:
        [Dataframe]: contains the two new columns pageviews and score.
    """

    # Add the new column pageviews
    quotes_score = get_pageview_quotes(quotes, speakers_pageviews)

    # Apply a min/max normalization on pageviews
    quotes_score['score'] = (quotes_score.pageviews - quotes_score.pageviews.min()) / (quotes_score.pageviews.max() - quotes_score.pageviews.min())

    # Return the final dataframe
    return quotes_score


# ----------------------------------------------------------------- #


def get_speakers_pageviews_per_year(speakers_labels):
    """
    This function add a new column for every year beween 2015 and 2020
    where there is the number of pageviews at the corresponding year 
    for the speaker's wikipage.

    Args:
        speakers_labels (dataframe): containing the speakers' labels.

    Returns:
        dataframe: new dataframe with all the new columns year.
    """

    # Copy the dataframe
    speakers_pageviews = speakers_labels.copy()

    # Loop along the years between 2015 and 2020
    for year in range(2015, 2021):
        # Add the new column for every year
        speakers_pageviews[str(year)] = speakers_pageviews.label.apply(lambda label: get_page_views_per_year(label, year))

    # Return the final dataframe
    return speakers_pageviews


# ----------------------------------------------------------------- #


def get_sentiment_quotes(quotes):
    """
    This function add a new column sentiment to our dataframe 
    representing the valence of each quotes as a value among {-1, 0, +1}.

    Args:
        quotes (dataframe): contains all the quotes

    Returns:
        [dataframe]: new dataframe with added column sentiement
    """

    # Copy the dataframe
    quotes_sentiment = quotes.copy()

    # Add the new column
    quotes_sentiment['sentiment'] = quotes_sentiment.quotation.progress_apply(sentiment_binary)

    # Return the dataframe
    return quotes_sentiment


# ----------------------------------------------------------------- #


def pos_score(row):
    """
    This function return the score (in positive value) if the sentiment
    is positive (ie equal to +1).

    Args:
        row (row of dataframe): row that contains the sentiment of the
        quote.

    Returns:
        [integer]: return the score.
    """

    # Set the score equal to zero
    score = 0

    # Update the score if the sentiment is positive
    if row.sentiment > 0:
        score = row.score

    # Return the result
    return score


# ----------------------------------------------------------------- #


def neg_score(row):
    """
    This function return the score (in negative value) if the sentiment
    is negative (ie equal to -1).

    Args:
        row (row of dataframe): row that contains the sentiment of the
        quote.

    Returns:
        [integer]: return the score.
    """

    # Set the score equal to zero
    score = 0

    # Update the score if the sentiment is negative
    if row.sentiment < 0:
        score = row.score

    # Return the result
    return score


# ----------------------------------------------------------------- #


def get_neg_pos_score_quotes(quotes):
    """
    This function add the new columns negative and positive score for
    every quotes.

    Args:
        quotes (dataframe): contains all the quotes with the score and
        sentiment.

    Returns:
        [Dataframe]: new dataframe with columns negative and positive 
        score.
    """

    # Copy the dataframe
    quotes_pos_neg_score = quotes.copy()

    # Add the new columns for positive and negative score
    quotes_pos_neg_score['positive_score'] = quotes_pos_neg_score.apply(pos_score, axis = 1)
    quotes_pos_neg_score['negative_score'] = quotes_pos_neg_score.apply(neg_score, axis = 1)

    # Return the new dataframe
    return quotes_pos_neg_score


# ----------------------------------------------------------------- #


def get_score_date(quotes):
    """
    This function sum all the positive and negative score for every
    days. It will be the dataframe used for the final plot.

    Args:
        quotes (dataframe): contains positive / negative score

    Returns:
        [dataframe]: new dataframe with positive / negative score
        by days.
    """

    # Copy the dataframe 
    score_date = quotes.copy()

    # Groupby days and keep the only columns we want
    score_date = score_date[['date', 'positive_score', 'negative_score']].dropna().groupby(['date']).sum()

    # Reset indices and sort in incresing order for days
    score_date = score_date.reset_index(drop = False)
    score_date = score_date.sort_values(by="date")
    score_date = pd.DataFrame(score_date.groupby(score_date.date.dt.date).sum())
    score_date = score_date.reset_index(drop = False)

    # Return the final dataframe
    return score_date


# ----------------------------------------------------------------- #


def correlation_stock_fame(score_date, stock):
    """
    This function computes the correlation coefficients for negative 
    and positive score against stock liquidity.

    Args:
        score_date (dataframe): contains postive / negative score 
        per days.
        stock (dataframe): contains liquidity of the Apple stock
    """

    # Copy the dataframe
    pos_per_day = score_date.copy()

    # Correlation positive score
    pos_per_day = pos_per_day.drop('negative_score', axis = 1)
    pos_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    pos_per_day['Date']= pd.to_datetime(pos_per_day['Date'], errors='coerce')
    stock_to_keep_pos = stock[stock.Date.isin(set(stock.Date).intersection(set(pos_per_day.Date)))]
    pos_per_day_to_keep = pos_per_day[pos_per_day.Date.isin(set(stock.Date).intersection(set(pos_per_day.Date)))]

    # Print the pearson
    print("Pearson positive score :", pearsonr(stock_to_keep_pos.Liquidity, pos_per_day_to_keep.positive_score))

    # Copy the dataframe
    neg_per_day = score_date.copy()

    # Correlation negative score
    neg_per_day = neg_per_day.drop('positive_score', axis = 1)
    neg_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    neg_per_day['Date']= pd.to_datetime(neg_per_day['Date'], errors='coerce')
    stock_to_keep_neg = stock[stock.Date.isin(set(stock.Date).intersection(set(neg_per_day.Date)))]
    neg_per_day_to_keep = neg_per_day[neg_per_day.Date.isin(set(stock.Date).intersection(set(neg_per_day.Date)))]
    
    # Print the pearson
    print("Pearson negative score :", pearsonr(stock_to_keep_neg.Liquidity, neg_per_day_to_keep.negative_score))


# ----------------------------------------------------------------- #