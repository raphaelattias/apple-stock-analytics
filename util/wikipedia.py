# This fil contains all the function we will use in the process of
# the wikipedia data set.

# ----------------------------------------------------------------- #


# Import all the libraries
import os
import pandas as pd
import numpy as np
import pageviewapi 
from tqdm import tqdm
from util.sentiment_analysis import *
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import iplot

from util.dataloader import *

tqdm.pandas()

# ----------------------------------------------------------------- #


# This function is use to transform all the qids in the datafram to 
# a string fromat.
def pandas_process(input):
    output = str(input)
    return output


# ----------------------------------------------------------------- #

# This function is here to get all the ID of the different speakers in 
# the wiki data set. The idea is to have list of all the person that
# have spoken in Quotebank da  ta set and keep only the wikipedia 
# information we want. 
def get_speakers_ids(quotes):
    """


    Args:
        quotes (pd.Dataframe): 

    Returns:
        (pd.Dataframe): 
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

# The idea of this function is to concatenate all the wiki files
# for the speaker attributes in one big file. The goal is not to keep
# a big file and keep surching for information inside, we will do a
# filtering with all the qids of all the speakers we have.
def get_wiki_labels():
    
    # Initialization of the files variable
    files = []

    # Initailize the category
    category = 'wiki speakers attributes'

    # Initialize the path
    path = 'data/wiki_speaker_attributes/'
    path = os.path.join(os.getcwd(), path)

    # Get the dictionniary conatining all the adresses
    wiki_dict = get_dictionnary()[category]

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



# The idea pf this function is to remov eall the duplicated line we 
# could have in our dataframe.
def remove_duplicates(df, column_name):

    # Intializa ou cleaned dataframe
    df_cleaned = df.copy()

    # Apply a string filter and drop the duplicates
    df_cleaned[column_name] = df_cleaned[column_name].apply(pandas_process).astype('|S')
    df_cleaned = df_cleaned.drop_duplicates([column_name])

    # Return the cleaned dataframe
    return df_cleaned



# ----------------------------------------------------------------- #



def get_page_views_per_year(page, year):

    nb_page_views = None

    if str(page) != 'None':
        # Time line
        begin = str(year) + '0101'
        end = str(year) + '1231'

        # Get the data frame from wikipedia API
        try:
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
    page_views = np.array([get_page_views_per_year(page, year) for year in range(wiki_page_year(page), 2021)])
    total_nb = np.sum(page_views)
    return total_nb



# ----------------------------------------------------------------- #


def exist_wiki_page_for_year(speaker, year):

    try :
        get_page_views_per_year(speaker, year)
        return True

    except:
        return False


# ----------------------------------------------------------------- #



def wiki_page_year(speaker):

    # Check wether this speaker has a wiki page
    for year in range(2015, 2021):
        if exist_wiki_page_for_year(speaker, year):
            return year

    # Return flase if it has not one
    return 2021

        

# ----------------------------------------------------------------- #



def exist_wiki_page(speaker):

    # Check wether this speaker has a wiki page
    for year in range(2015, 2021):
        if exist_wiki_page_for_year(speaker, year):
            return True

    # Return flase if it has not one
    return False


# ----------------------------------------------------------------- #


def wiki_label_speaker(ids, wiki_data):

    # Initialization
    label = None
    prev_nb_views = 0

    # Go through all the qids for a specific speaker
    for id in ids:
        try:
            speaker = wiki_data.label[wiki_data.id == id].reset_index(drop = True).values[0]
            total_page_views = get_total_page_views(speaker)

            if exist_wiki_page(speaker) and total_page_views > prev_nb_views:
                label = speaker
                prev_nb_views = total_page_views
        except:
            pass
        
    return label

# ----------------------------------------------------------------- #


def save_speakers_id(speakers_id, save, id_nb):
    if save:
        name = 'speakers_labels_' + str(id_nb)
        save_path = os.path.join('data/wiki_speaker_attributes', name+ '.pkl')
        if os.path.isfile(save_path):
            print(f"WARNING: the file {save_path} already exists and will be deleted at the end.")
        speakers_id.to_pickle(save_path)

    return None




# ----------------------------------------------------------------- #


def get_speakers_labels():
    folder_path = 'data/wiki_speaker_attributes/'
    file_name = 'speakers_labels_'
    speakers_labels = pd.read_pickle(os.path.join(folder_path + file_name + '1.pkl'))

    for idx_file in range(1, 21):
        path = os.path.join(folder_path + file_name + str(idx_file) + '.pkl')
        current_speakers_labels = pd.read_pickle(path)
        speakers_labels = pd.concat([speakers_labels, current_speakers_labels])
    
    speakers_labels = speakers_labels.drop_duplicates(subset=['speaker']).reset_index(drop = True)

    return speakers_labels


# ----------------------------------------------------------------- #


def get_speakers_labels_one_file():
    folder_path = 'data/wiki_speaker_attributes/'
    file_name = 'speakers_labels.pkl'
    speakers_labels = pd.read_pickle(os.path.join(folder_path + file_name))
    
    speakers_labels = speakers_labels.drop_duplicates(subset=['speaker']).reset_index(drop = True)

    return speakers_labels


# ----------------------------------------------------------------- #

def find_labels(speakers_id, wiki_data):
    speakers_id_new = speakers_id.copy()
    speakers_label = speakers_id.qids.progress_apply(lambda ids: wiki_label_speaker(ids, wiki_data))
    speakers_id_new['label'] = speakers_label

    return speakers_id_new


# ----------------------------------------------------------------- #


def add_labels(speakers_id, wiki_data, save = False, cluster = None):

    if cluster == None:
        speakers_id_new_df = speakers_id.copy()
        nb_elem = speakers_id_new_df.shape[0]
        batch_size = nb_elem / 20

        speakers_id_batch = speakers_id.copy().iloc[np.arange(0, batch_size).astype(int)]

        speakers_id_batch = find_labels(speakers_id, wiki_data)

        save_speakers_id(speakers_id_batch, save, 1)

        for batch in range(1, 20):
            batch_begin = (batch * batch_size) + 1
            batch_end = min(nb_elem, batch_begin + batch_size)

            speakers_id_batch = speakers_id.copy().iloc[np.arange(batch_begin, batch_end).astype(int)]

            speakers_id_batch = find_labels(speakers_id_batch, wiki_data)

            save_speakers_id(speakers_id_batch, save, batch + 1)

    else:
        speakers_id_new_df = speakers_id.copy()
        nb_elem = speakers_id_new_df.shape[0]
        sub_set_size = nb_elem / 4

        begin = (cluster - 1) * sub_set_size
        if cluster == 4:
            end = nb_elem
        else:
            end = (cluster * sub_set_size) - 1

        sub_set_nb_elem = end - begin + 1

        batch_size = sub_set_nb_elem / 5

        for batch in range(0, 5):

            batch_begin = begin + (batch_size * batch)
            batch_end = min(end, batch_begin + batch_size - 1)

            speakers_id_batch = speakers_id.copy().iloc[np.arange(batch_begin, min(end, batch_end + 1)).astype(int)]

            speakers_id_batch = find_labels(speakers_id_batch, wiki_data)

            id_nb = ((cluster - 1) * 5) + 1 + batch

            save_speakers_id(speakers_id_batch, save, id_nb)
    
    speakers_id_new_df = get_speakers_labels()

    return speakers_id_new_df



# ----------------------------------------------------------------- #



def scoring(quotes):
    try:
        score = quotes[str(quotes['year'])]
    except:
        score = None
    return score


# ----------------------------------------------------------------- #

def get_pageview_quotes(quotes, speakers_pageviews):
    quotes_pageviews = quotes.copy()
    quotes_pageviews['year'] = quotes_pageviews.date.apply(lambda date: date.year)
    quotes_pageviews = quotes_pageviews.set_index('speaker').join(speakers_pageviews.set_index('speaker'), lsuffix="_left", rsuffix="_right").reset_index()
    quotes_pageviews = quotes_pageviews[quotes_pageviews.year >= 2015].copy()
    quotes_pageviews['pageviews'] = quotes_pageviews.apply(scoring, axis = 1)
    quotes_pageviews = quotes_pageviews.drop(['2015', '2016', '2017', '2018', '2019', '2020', 'year'], axis = 1).reset_index(drop = True)
    return quotes_pageviews

# ----------------------------------------------------------------- #


def get_score_quotes(quotes, speakers_pageviews):
    quotes_score = get_pageview_quotes(quotes, speakers_pageviews)
    quotes_score['score'] = (quotes_score.pageviews - quotes_score.pageviews.min()) / (quotes_score.pageviews.max() - quotes_score.pageviews.min())

    return quotes_score



# ----------------------------------------------------------------- #


def get_speakers_pageviews_per_year(speakers_labels):
    speakers_pageviews = speakers_labels.copy()
    for year in range(2015, 2021):
        speakers_pageviews[str(year)] = speakers_pageviews.label.apply(lambda label: get_pageviews_per_year(label, year))
    return speakers_pageviews

# ----------------------------------------------------------------- #



def get_pageviews_per_year(page, year):

    nb_page_views = None

    if str(page) != 'None':
        # Time line
        begin = str(year) + '0101'
        end = str(year) + '1231'

        # Get the data frame from wikipedia API
        try:
            df_wiki_api = pd.DataFrame(pageviewapi.per_article('en.wikipedia', page, begin, end,
                            access='all-access', agent='all-agents', granularity='monthly')['items'])
        
            # Get the number of page views
            nb_page_views = df_wiki_api.views.sum()
        except:
             pass

    # Return the results
    return nb_page_views



# ----------------------------------------------------------------- #


def get_sentiment_quotes(quotes):
    quotes_sentiment = quotes.copy()
    quotes_sentiment['sentiment'] = quotes_sentiment.quotation.progress_apply(sentiment_binary)
    return quotes_sentiment


# ----------------------------------------------------------------- #

def pos_score(row):
    score = 0
    if row.sentiment > 0:
        score = row.score
    return score

# ----------------------------------------------------------------- #

def neg_score(row):
    score = 0
    if row.sentiment < 0:
        score = -row.score
    return score


# ----------------------------------------------------------------- #


def get_neg_pos_score_quotes(quotes):
    quotes_pos_neg_score = quotes.copy()
    quotes_pos_neg_score['positive_score'] = quotes_pos_neg_score.apply(pos_score, axis = 1)
    quotes_pos_neg_score['negative_score'] = quotes_pos_neg_score.apply(neg_score, axis = 1)
    return quotes_pos_neg_score


# ----------------------------------------------------------------- #


def get_score_date(quotes):
    score_date = quotes.copy()
    score_date = score_date[['date', 'positive_score', 'negative_score']].dropna().groupby(['date']).sum()
    score_date = score_date.reset_index(drop = False)
    score_date = score_date.sort_values(by="date")
    score_date = pd.DataFrame(score_date.groupby(score_date.date.dt.date).sum())
    score_date = score_date.reset_index(drop = False)
    return score_date


def stock_price_against_quotes_score(score_date, stock_all):
    stock_analysis = stock_all.copy()

    date_min = '2015-01-01'
    stock_analysis = stock_analysis[stock_analysis.Date >= date_min]

    date_max = '2020-04-16'
    stock_analysis = stock_analysis[stock_analysis.Date <= date_max]

    stock_analysis['stock_price'] = (stock_analysis.Open + stock_analysis.Low) / 2
    


    trace1 = go.Scatter(
        x = score_date.date,
        y = score_date.positive_score,
        mode = 'lines',
        name = 'Positive',
        marker=dict(color='rgb(30,50,130)')
    )

    trace2 = go.Scatter(
        x = score_date.date,
        y = score_date.negative_score,
        mode = 'lines',
        name = 'Negative',
        marker=dict(color='rgb(150,37,30)')
    )

    trace3 = go.Scatter(
        x = stock_analysis.Date,
        y = stock_analysis.stock_price,
        mode = 'lines',
        name = 'Stock',
        # marker=dict(color='rgb(250,125,62)'),
        marker=dict(color='rgb(25,125,35)'),
        opacity = 0.4
    )

    pio.renderers.default = "notebook_connected"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace3, secondary_y=True)
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig['layout'].update(height = 600, 
                            width = 800, 
                            title = 'Distribution of the positive and negative')
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

    max_quotes = max(-score_date.negative_score.min(), score_date.positive_score.max())
    y_max_quotes = max_quotes + (max(-score_date.negative_score.min(), score_date.positive_score.max())/10)
    y_min_quotes = -max_quotes + (max(-score_date.negative_score.min(), score_date.positive_score.max())/10)

    max_stock = max(-stock_analysis.stock_price.min(), stock_analysis.stock_price.max()) + (max(-stock_analysis.stock_price.min(), stock_analysis.stock_price.max()) / 10)
    y_max_stock = max_stock + (max(-stock_analysis.stock_price.min(), stock_analysis.stock_price.max()) / 10)
    y_min_stock = -max_stock + (max(-stock_analysis.stock_price.min(), stock_analysis.stock_price.max()) / 10)

    fig.update_xaxes(title_text = 'Date')
    fig.update_yaxes(range=[y_min_quotes, y_max_quotes], 
                        secondary_y=False, 
                        title_text = 'Score'
    )
    fig.update_yaxes(range=[y_min_stock, 
                        y_max_stock], 
                        secondary_y=True,
                        title_text = 'Stock price [$]'
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    fig.update_layout(bargap=0.1, bargroupgap = 0, template='ggplot2', )
    iplot(fig)

    fig.write_html('figures/stock_price_against_quotes_score.html')