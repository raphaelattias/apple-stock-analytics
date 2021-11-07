import json
import bz2
import pandas as pd
import os
from tqdm.notebook import tqdm
from tqdm import trange
import gdown

# conda install -c conda-forge ipywidgets
# conda install tqdm
# jupyter nbextension enable --py widgetsnbextension


def download(path):
    """Download the respective dataset from Gdrive

    Args:
        path (string): Path to the dataset.
    """

    filename = os.path.split(path)[-1]
    files = { \
        'quotes-2008.json.bz2': '1wIdrR0sUGw7gAKCo_S-iL3q_V04wHzrP', \
        'quotes-2009.json.bz2': '1Wds32frDJ6PJgP1ruU2ctDvvlcOF4k3i', \
        'quotes-2010.json.bz2': '1dUMLpB7rVRF3nY6X2GmVNO57Zm1RVZRB', \
        'quotes-2011.json.bz2': '1sPlhxtt9VJROcaD97DmzHsFROGBOCpK6', \
        'quotes-2012.json.bz2': '1M3arwVzCNz9n92wJVl9c3rTOU5oh1xFQ', \
        'quotes-2013.json.bz2': '1PZEmS85TAHtNwXoMgm-7MDC58oS3cK73', \
        'quotes-2014.json.bz2': '1axK0PRItbbQW4V-T1fDa3J75bKZJHVLI', \
        'quotes-2015.json.bz2': '1ujF5vgppXUu5Ph81wqrwY12DrszVmCGe', \
        'quotes-2016.json.bz2': '1iyYhemohtPBwFyWck8SMHdaHoJMZShsI', \
        'quotes-2017.json.bz2': '1823mXyPsLDK7i1CQ7CtjzJaJ8rxeEulp', \
        'quotes-2018.json.bz2': '1X609SehGUxgoB0LfwazAeySjWDc-VhcZ', \
        'quotes-2019.json.bz2': '1KUXgpssbM7mXGx5RqturDKdtdS_KxIB8', \
        'quotes-2020.json.bz2': '1kBPm86V1_9z-9rTi3F-ENgxGvUod0olI'}
    url = f'https://drive.google.com/uc?id={files[filename]}'
    gdown.download(url, path, quiet=False)

def filter_quotes(path, keywords = [""], speakers = [""], chunksize = 1000, save = None, chunknum = None):
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
    assert (keywords != [""] and speakers !=  [""], "The keywords and speakers are empty lists, nothing to filter.")

    if save != None:
        save_path = os.path.join("data/processed/", save+'.pkl')
        if os.path.isfile(save_path):
            print(f"WARNING: the file {save_path} already exists and will be deleted at the end.")
    path = os.path.join(os.getcwd(),path)
    if not os.path.isfile(path):
        download(path)

    tp = pd.read_json(path, lines= True, chunksize=chunksize)

    if chunknum != None:
        iterator = tqdm(tp, total=chunknum, desc="Chunks filtered", unit = "chunk", smoothing=0.6)
        print(f"INFO: {chunknum*chunksize} quotes will be inspected")
    else:
        iterator = tqdm(tp)

    for num, chunk in enumerate(iterator):
        if num == chunknum:
            break

        df_temp = pd.DataFrame(chunk, columns=chunk.keys())
        df_temp = df_temp[df_temp['speaker'].str.contains('|'.join(speakers)) \
                | df_temp["quotation"].str.split(" ").apply(lambda x : bool(set(x) & set(keywords)))]
        if num == 0:
            df = df_temp
        else:
            df = df.append(df_temp, ignore_index= True)

    if save != None:
        df.to_pickle(save_path)

    return df

def load_quotes(path, limit = None, columns = None, low_memory = False):
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

    path = os.path.join(os.getcwd(),path)
    if not os.path.isfile(path):
        download(path)


    with bz2.open(path, "rt", encoding = "utf8") as bzinput:
        quotes = []
        for i, line in enumerate(bzinput):
            if limit != None and i == limit: break

            quote = json.loads(line)

            if columns == None:
                columns = list(quote.keys())
                
            new_quote = []
            for col in columns:
                new_quote.append(quote[col])
            quotes.append(new_quote)

    return pd.DataFrame(quotes,columns=columns)
