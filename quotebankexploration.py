from IPython.display import display
import matplotlib.pyplot as plt

def quotebank_exploration(quote_df) : 
    #give an overview of quote_df
    print("Let's see what the dataset looks like and what is it's shape.")
    display(quote_df.head(5))
    display(quote_df.shape)
    
def refilter(quote_df):
    #set every speaker's name to lowercase to avoid duplicates 
    quote_df["speaker"] = quote_df["speaker"].apply(lambda x : str(x).lower())

    #refilter the quotebank by removing words we don't want to appear in our final dataset
    quote_df = quote_df[~quote_df['quotation'].str.contains("freddie mac")]
    quote_df = quote_df[~quote_df['quotation'].str.contains("johnny mac")]
    quote_df = quote_df[~quote_df['quotation'].str.contains("big mac")]

    return quote_df

def plot_table_numOcc(quote_df, head_ = 10):
    df_table = quote_df.groupby("speaker").sum().sort_values(by=["numOccurrences"],ascending = False).drop(["none"]).head(head_)
    return df_table
