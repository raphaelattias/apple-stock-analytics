# This fil contains all the function we will use in the process of
# the wikipedia data set.

# ----------------------------------------------------------------- #


# Import all the libraries
import os
import pandas as pd
import numpy as np
import pageviewapi 
from tqdm import tqdm

from dataloader import *


# ----------------------------------------------------------------- #


# This function is use to transform all the qids in the datafram to 
# a string fromat.
def pandas_process(input):
    output = str(input)
    return output


# ----------------------------------------------------------------- #

''' !!! TO SUPRESS !!! '''

# This function is here to get all the ID of the different speakers in 
# the wiki data set. The idea is to have list of all the person that
# have spoken in Quotebank da  ta set and keep only the wikipedia 
# information we want. 
def get_speakers_ids(quotes):

    # Copy the data frame
    wiki_quotes = quotes.copy()
    all_indices = wiki_quotes.index

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
def concat_wiki_files():
    
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
    tqdm.pandas

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