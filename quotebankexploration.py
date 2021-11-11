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

    #refilter the quotebank by removing words we don't want to appear in our dataset
    quote_df = quote_df[~quote_df['quotation'].str.contains("freddie mac")]
    quote_df = quote_df[~quote_df['quotation'].str.contains("johnny mac")]
    quote_df = quote_df[~quote_df['quotation'].str.contains("big mac")]

    return quote_df

def plot_table_numOcc(quote_df, head_ = 10):
    df_table = quote_df.groupby("speaker").sum().sort_values(by=["numOccurrences"],ascending = False).drop(["none"]).head(head_)
    return df_table

def plot_pie_numquote(quote_df, year, head_ = 5):
    ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    ax = quote_df.groupby("speaker")["quotation"].count().sort_values(ascending = False).drop(["none"]).head(head_).plot.pie()
    ax.set(ylabel = None)
    #ax.legend(title="Speakers", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title("Repartition of the number of quotes between the %i most commun speakers" %head_ + " for the year %i"  %year )
    plt.show()


def plot_quotes_per_day(quote_df, year):
    quote_df.groupby("date")["quotation"].count().plot()
    plt.title('Numbers of quotes per day in %i' %year)
    plt.show()

def plot_numOcc_per_day(quote_df, year):
    quote_df.groupby("date").sum().plot()
    plt.title('Numbers of occurrences of all the quotes per day in %i' %year)
    plt.show()
