import json
import bz2
import pandas as pd
import os
from tqdm.notebook import tqdm
from tqdm import trange
import gdown
from util.quotebankexploration import *

# conda install -c conda-forge ipywidgets
# conda install tqdm
# jupyter nbextension enable --py widgetsnbextension


# ----------------------------------------------------------------- #


def download(path, category):
    """Download the respective dataset from Gdrive

    Args:
        path (string): Path to the dataset.
        category (string):
    """

    # Get the file name of what we are looking for
    filename = os.path.split(path)[-1]

    # Get the dictionnary of all the file adresses from our google drive
    files = get_drive_dictionnary()


    url = f'https://drive.google.com/uc?id={files[category][filename]}'
    gdown.download(url, path, quiet=False)


# ----------------------------------------------------------------- #


def filter_quotes(path, keywords = {}, speakers = [""], chunksize = 1000, save = None, chunknum = None):
    """Filter a compressed dataset of quotes, with respect to some keywords and speakers. This function
    can be used to save the filtered heavy dataset into a light pickle file, by specifing a string name
    for the argument save.

    Example:
    obama_quotes = filter_quotes(r"data/quotes-2008.json.bz2",\
        speakers = ['obama'], \
        keywords=["yes we can"], \
        chunknum = 1000 \
        save = "obama-quotes")

    Args:
        path (Path): Path to the compressed json.bz2 file of quotes
        keywords (list, optional): List of keywords for filtering the quotations. Defaults to [""].
        speakers (list, optional): List of speakers to filter the speaker column. Defaults to [""].
        chunksize (int, optional): Size of the chunks to be loaded in memory. Defaults to 1000.
        save (string, optional): Name of the filtered panda dataframe to be saved. Defaults to None.
        chunknum (int, optional): Maximum number of chunks to load. Defaults to None.

    Returns:
        pd.dataframe: Filtered panda dataframe
    """
    
    if save != None:
        save_path = os.path.join("data/processed_quotes/", save+'.pkl')
        if os.path.isfile(save_path):
            print(f"WARNING: the file {save_path} already exists and will be deleted at the end.")
    path = os.path.join(os.getcwd(),path)
    if not os.path.isfile(path):
        download(path, 'unprocessed quotes')

    tp = pd.read_json(path, lines= True, chunksize=chunksize)

    if chunknum != None:
        iterator = tqdm(tp, total=chunknum, desc="Chunks filtered", unit = "chunk", smoothing=0.6)
        print(f"INFO: {chunknum*chunksize} quotes will be inspected")
    else:
        iterator = tqdm(tp)
 
    total_nb = 0
    for num, chunk in enumerate(iterator):
        if num == chunknum:
            break

        # The idea here is to split the filter in two parts, where we want precise unique words in a quote,
        # and secondly we want a precise group of words (e.g. two words) in our quotes. This is done in the
        # following two steps.

        df_temp = pd.DataFrame(chunk, columns=chunk.keys())

        total_nb += len(df_temp)

        criteria_speakers = df_temp['speaker'].apply(lambda x : x.lower()).str.contains('|'.join(speakers))
        criteria_1 = df_temp["quotation"].apply(lambda x : x.lower()).str.split(" ").apply(lambda x : bool(set(x) & set(keywords["One word"])))
        criteria_2 = df_temp["quotation"].apply( \
            lambda x : bool(set(map(' '.join, zip(*(x.lower().split(" ")[i:] for i in range(2))))) \
             & set(keywords["Two words"])))
        criteria_3 = df_temp["quotation"].str.split(" ").apply(lambda x : bool(set(x) & set(keywords["Capital words"])))
        criteria_black_list = df_temp["quotation"].apply( lambda x : bool(set(map(' '.join, zip(*(x.split(" ")[i:] for i in range(2))))) \
             & set(keywords["Black list"])))
    
        df_temp = df_temp[(criteria_speakers | criteria_1 | criteria_2 | criteria_3) & (~criteria_black_list)]

        if num == 0:
            df = df_temp
        else:
            df = df.append(df_temp, ignore_index= True)

    if save != None:
        df.to_pickle(save_path)
    
    print(f"INFO: {len(df)} citations have been kept over ", total_nb, " total number of citations.")

    return {"dataframe": df, "total": total_nb}


# ----------------------------------------------------------------- #


def load_quotes(year, category, limit = None, columns = None):
    """Function to load the quotes of a compressed json file into a pd.DataFrame

    Args:
        path (Path): Path to the compressed json.bz2 file of quotes
        limit (Int, optional): Maximum number of item to load . Defaults to None.
        cols (List[String], optional): Columns to load into the dataframe. Defaults to None.

    Returns:
        pd.Datframe: Panda Dataframe that contains the limit-first quotes with the colums cols.

    Remark: 
        Originally all the columns are
            ['quoteID', 'quotation', 'speaker', 'qids', 'date', 'numOccurences', 'probas', 'urls', 'phase']
    """

    if category == 'processed quotes':
        path = os.path.join(os.getcwd(),'data/processed_quotes/',f"filtered_quotes_{str(year)}.pkl")
    elif category == 'unprocessed quotes':
        path = os.path.join(os.getcwd(),'data/unprocessed_quotes/',f"quotes-{str(year)}.json.bz2")
    else:
        print('ERROR: For this load_quotes function, category variable can be either \
                         -- unprocessed quotes -- or -- processed quotes --. ')

    if not os.path.isfile(path):
        download(path, category)

    if category == 'processed quotes':
        df = pd.read_pickle(path)
    elif category == 'unprocessed quotes':
        with bz2.open(path, "rt", encoding = "utf8") as bzinput:
            quotes = []
            for i, line in tqdm(enumerate(bzinput)):
                if limit != None and i == limit: break

                quote = json.loads(line)

                if columns == None:
                    columns = list(quote.keys())
                    
                new_quote = []
                for col in columns:
                    new_quote.append(quote[col])
                quotes.append(new_quote)
        df = pd.DataFrame(quotes,columns=columns)
    return df


# ----------------------------------------------------------------- #


# The idea here is to concatenated in one single dataframe all the
# filtered quotes.
def get_filtered_quotes():

    # Initialize a list
    filtered_quotes = []

    # Initialize the category
    category = 'processed quotes'

    # Initialize the path
    path = 'data/processed_quotes/'
    path = os.path.join(os.getcwd(), path)

    # Get the filtered quotes dictionnary
    filt_quotes_dict = get_drive_dictionnary()[category]

    # Get a loop over all the filtered quotes
    for key in filt_quotes_dict:
        
        # Get the path where the pkl filtered quotes file is
        current_path = os.path.join(path, key)

        # Check wether the file is download or not
        if not os.path.isfile(current_path):
            download(current_path, category)

        # Append in the list the current filtered quotes
        filtered_quotes.append(pd.read_pickle(current_path))

    # Concatenation
    filtered_quotes = pd.concat(filtered_quotes)

    # Refilter
    filtered_quotes = refilter(filtered_quotes)

    filtered_quotes = filtered_quotes.reset_index(drop = True)

    # Return the results
    return filtered_quotes



# ----------------------------------------------------------------- #


def get_speakers_pageviews():

    # Initialize the path
    path = 'data/wiki_speaker_attributes/speakers_pageviews.pkl'
    path = os.path.join(os.getcwd(), path)

    # Set the category
    category = 'speakers attributes'

    if not os.path.isfile(path):
        download(path,category)

    speakers_pageviews = pd.read_pickle('data/wiki_speaker_attributes/speakers_pageviews.pkl')

    return speakers_pageviews


# ----------------------------------------------------------------- #


def get_speakers_labels():

    # Initialize the path
    path = 'data/wiki_speaker_attributes/speakers_labels.pkl'
    path = os.path.join(os.getcwd(), path)

    # Set the category
    category = 'speakers attributes'

    if not os.path.isfile(path):
        download(path, category)

    speakers_labels = pd.read_pickle('data/wiki_speaker_attributes/speakers_labels.pkl')

    return speakers_labels


# ----------------------------------------------------------------- #



def get_drive_dictionnary():

    # Dictionnary
    files = {
        'unprocessed quotes': {
            'quotes-2008.json.bz2': '1wIdrR0sUGw7gAKCo_S-iL3q_V04wHzrP',
            'quotes-2009.json.bz2': '1Wds32frDJ6PJgP1ruU2ctDvvlcOF4k3i',
            'quotes-2010.json.bz2': '1dUMLpB7rVRF3nY6X2GmVNO57Zm1RVZRB',
            'quotes-2011.json.bz2': '1sPlhxtt9VJROcaD97DmzHsFROGBOCpK6',
            'quotes-2012.json.bz2': '1M3arwVzCNz9n92wJVl9c3rTOU5oh1xFQ',
            'quotes-2013.json.bz2': '1PZEmS85TAHtNwXoMgm-7MDC58oS3cK73',
            'quotes-2014.json.bz2': '1axK0PRItbbQW4V-T1fDa3J75bKZJHVLI',
            'quotes-2015.json.bz2': '1ujF5vgppXUu5Ph81wqrwY12DrszVmCGe',
            'quotes-2016.json.bz2': '1iyYhemohtPBwFyWck8SMHdaHoJMZShsI',
            'quotes-2017.json.bz2': '1823mXyPsLDK7i1CQ7CtjzJaJ8rxeEulp',
            'quotes-2018.json.bz2': '1X609SehGUxgoB0LfwazAeySjWDc-VhcZ',
            'quotes-2019.json.bz2': '1KUXgpssbM7mXGx5RqturDKdtdS_KxIB8',
            'quotes-2020.json.bz2': '1kBPm86V1_9z-9rTi3F-ENgxGvUod0olI'
        },
        'processed quotes': {
            'filtered_quotes_2008.pkl': '1pmP2oz9S5W2t0ILVlUn27D9Ad3zSSTaS',
            'filtered_quotes_2009.pkl': '1U7bj4XTR9TAXckTBk7LnwVmk2_RLcIDg',
            'filtered_quotes_2010.pkl': '1PykPkem69dAhzsfqZJeoC48XDEIsWlnA',
            'filtered_quotes_2011.pkl': '1M07hp-Pxqab3lwxs_2eUiR3YZGZSnmiO',
            'filtered_quotes_2012.pkl': '1stTLHJeY_W48L9QPuJuYD-4MHIlahKLa',
            'filtered_quotes_2013.pkl': '1LyNfF6M6QK7G8n-mYO_-RJ6QPOHVFGqn',
            'filtered_quotes_2014.pkl': '13fk6lVH3kOwCI5t_HiiByozcMipMszsD',
            'filtered_quotes_2015.pkl': '1WJ2D2RJrcR67KeDUzKzGqKourxyX2V3u',
            'filtered_quotes_2016.pkl': '1SHUWidmpLJUdbD3uoPv1RT4TBw6_ID4H',
            'filtered_quotes_2017.pkl': '1KNRHJvJyjZMSQm4F98rGs18zNnvxNdBR',
            'filtered_quotes_2018.pkl': '1burATSHOF-bLgZmwY09upe9DcMPqgD0s',
            'filtered_quotes_2019.pkl': '1gr4Tk5WlOB_n-2i409XjWCSEIzvmsrDz',
            'filtered_quotes_2020.pkl': '1ihiwu0nMJJCSCXQLyXZTyuY0YfR5ZY0i'
        }, 
        'wiki speakers attributes': {
            'part-00000-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1S5EJgUwjw8QknkjUAjEO7FUvIQQZXong',
            'part-00001-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1qAr9dICWbkEtzTx9jyseg8p1gMd-5qQz',
            'part-00002-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '12MPMK63m5Xa4XM5D360Hg315wfSUu9pn',
            'part-00003-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '13y6Oh6s6FEOnmehNKqRULBekDfRHffYH',
            'part-00004-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1L2ci2rrMPNuVmoOwZ9pYkmwc9w4MEn_M',
            'part-00005-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '16bqF0FuoV0QLG7vYgeR638Q4gINMcI42',
            'part-00006-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '10tIX5WCxyBaZIEm3WgoK1uKIt5RQ1nkZ',
            'part-00007-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1JhqpUiPUWwdTRIv-rbn0IaG0Y1j5SaJJ',
            'part-00008-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1IKbPyYxyw8Lewe9Q6ifWVsKneihUIpdW',
            'part-00009-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1noGdzogNAvgEQYgBO-c0-CO-jSpD-upp',
            'part-00010-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1POA0IgDCv7bJLNp7vdfGBxucEYQWoy68',
            'part-00011-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1fYRTQRmvrIwQigOMPqukW6_LMpq74AaU',
            'part-00012-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1lskprayUi9mB1U12fDpd2wsklwhV4g5r',
            'part-00013-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '15d1Xb7aLhQ_tJ_Mc9O5O5NuRlkvZd1Ar',
            'part-00014-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1ciVEdCJdJ-ymSpEr1x9Xg3yQvXaug56k',
            'part-00015-0d587965-3d8f-41ce-9771-5b8c9024dce9-c000.snappy': '1DSYYRitpC3NwEL0S5uijas0QRXrH505E',
        },
        'speakers attributes': {
            'speakers_pageviews.pkl' : '1dFU-92I-upV9BMhMryGBvYmmpwAgM2Sk',
            'speakers_labels.pkl': '1FIvRNItqIW5s7fPKOoZFGCxD6keBi1av'
        },
    }

    # Return the final files
    return files