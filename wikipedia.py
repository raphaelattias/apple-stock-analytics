# This fil contains all the function we will use in the process of
# the wikipedia data set.

# ----------------------------------------------------------------- #


# Import all the libraries
import os
import pandas as pd
import numpy as np
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
# the wiki data set. The idea is to have list of all the personb that
# have spoken in Quotebank data set and keep only the wikipedia 
# information we want. 
def get_wiki_ids(quotes):

    # Copy the data frame
    quotes_ID = quotes.copy()

    # Apply the string filter and drop all the duplicates
    quotes_ID['qids'] = quotes_ID['qids'].apply(pandas_process).astype('|S')
    quotes_ID = quotes_ID.drop_duplicates(['qids'])

    # Get the list of ID's
    quotes_ID = quotes_ID['qids']

    # Return the result
    return quotes_ID


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