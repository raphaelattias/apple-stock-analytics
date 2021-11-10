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
    wiki_ids = quotes_ID['qids']

    return wiki_ids


# ----------------------------------------------------------------- #

# The idea of this function is to concatenate all the wiki files
# for the speaker attributes in one big file. The goal is not to keep
# a big file and keep surching for information inside, we will do a
# filtering with all the qids of all the speakers we have.
def concat_wiki_files():
    
    # Initialization of the files variable
    files = []
    category = 'wiki speakers attributes'

    # Initialize the path
    path = 'data/wiki_speaker_attributes/'
    path = os.path.join(os.getcwd, path)

    # Get the dictionniary conatining all the adresses
    wiki_dict = get_dictionnary()['wiki speakers attributes']

    # Loop over all the wiki files containing speaker attributes
    for key in wiki_dict:
        print(key)
        current_path = os.path.join(path, key)
        if not os.path.isfile(current_path):
            download(current_path, category)
        files.append(pd.read_parquet(current_path))


    df_wiki = pd.concat(files)

    return df_wiki