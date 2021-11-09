# This fil contains all the function we will use in the process of
# the wikipedia data set.

# ----------------------------------------------------------------- #

# Import all the libraries
import pandas as pd
import numpy as np

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
def get_wiki_ids(quotes_ID):

    # Copy the data frame
    # quotes_ID = quotes.copy()

    # Apply the string filter and drop all the duplicates
    quotes_ID['qids'] = quotes_ID['qids'].apply(pandas_process).astype('|S')
    quotes_ID = quotes_ID.drop_duplicates(['qids'])

    # Get the list of ID's
    wiki_ids = quotes_ID['qids']

    return wiki_ids


# ----------------------------------------------------------------- #