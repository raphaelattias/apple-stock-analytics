# conda install pyarrow fastparquet pandas gdown

# Imports you may need
from dataloader import *
from finance import *
import pandas as pd
import os

from quotebankexploration import *
from wikipedia import *

if __name__ == "__main__":
    # Get some data frames
    wiki_data = concat_wiki_files()
    filtered_quotes = get_filtered_quotes()
    speakers_id = pd.DataFrame(get_speakers_ids(filtered_quotes))
    speakers_labels = get_speakers_labels()

    '''
    REMARK: Tu marques une année pour chaque cluster
    '''
    year = 2015

    # DO NOT TOUCH
    path = os.path.join('data/wiki_speaker_attributes/pageviews_'+str(year))
    speakers_labels = speakers_labels.label.apply(lambda label: get_page_views_per_year(label, year))
    speakers_labels.to_pickle(path)