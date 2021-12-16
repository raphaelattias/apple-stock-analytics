import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from tqdm import tqdm
from util.sentiment_analysis import *
from util.wikipedia import get_score_quotes
import plotly.graph_objs as go
from sklearn.utils import shuffle

alpha_ = 0.7
figsize = [10.0, 6.0]
DPI = 250

tqdm.pandas()

# ----------------------------------------------------------------- #


def bar_plots_quotes(frequency_all, frequency_apple, years):
    """
    Plot of the frequency of all the quotes from the quotebank compared to the quotes linked to Apple.

    Inputs:
        frequency_all (pd.Dataframe): frequency of all the quotes given in quotebank
        frequency_apple (pd.Dataframe): frequency of the filtered quotes only concerning Apple
        years (array<int>): array of all the years
    """
    barwidth = 0.3

    position_1 = np.arange(len(years)) +  (barwidth / 2)
    position_2 = np.arange(len(years)) + ((3 * barwidth) / 2)

    plt.figure(figsize = figsize)

    plt.bar(position_1, frequency_all, width = barwidth, label = 'All quotes', 
        alpha = alpha_)
    plt.bar(position_2, frequency_apple, width = barwidth, label = 'Apple quotes', 
        alpha = alpha_)

    years_str = [str(i) for i in years]

    plt.xticks([r + barwidth for r in range(len(years))], years_str)
    plt.yscale('log')
    plt.grid(True)
    plt.ylabel('Frequency')
    plt.ylabel('Years')

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.legend()

    return 0


# ----------------------------------------------------------------- #


def plot_pie_numquote(quote_df, year, head_ = 5):
    """
    Pie plot of the distribution of the repartition of the number of quotes between the most frequent speaker
        for a specific year.

    Inputs:
        quote_df (pd.Dataframe): Dataframe of quotes for the year chosen
        year (int): year of study
        head_ (int): number of most frequent speaker we want in the plot
    """
    ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    ax = quote_df.groupby("speaker")["quotation"].count().sort_values(ascending = False).drop(["none"]).head(head_).plot.pie()
    ax.set(ylabel = None)
    ax.set_title("Repartition of the number of quotes between the %i most commun speakers" %head_ + " for the year %i"  %year )

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.show()


# ----------------------------------------------------------------- #


def plot_quotes_per_day(quote_df, year):
    """
    Plot the distribution of the number of quotes group by day for a specific year.

    Inputs:
        quote_df (pd.Dataframe): Dataframe of quotes for the year chosen
        year (int): year of study
    """
    quote_df.groupby("date")["quotation"].count().plot(alpha = alpha_)
    plt.title('Numbers of quotes per day in %i' %year)
    plt.ylabel("Number of quotes")

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.show()


# ----------------------------------------------------------------- #


def plot_numOcc_per_day(quote_df, year):
    """
    Plot the distribution of the number of occurrences of the quotes group by day for a specific year.

    Inputs:
        quote_df (pd.Dataframe): Dataframe of quotes for the year chosen
        year (int): year of study
    """
    quote_df.groupby("date").sum().plot(alpha = alpha_)
    plt.title('Numbers of occurrences of all the quotes per day in %i' %year)
    plt.ylabel("Number of occurrences")

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.show()


# ----------------------------------------------------------------- #

def plot_wordcloud_speakers(quotes, speakers_pageviews, path = 'figures/wordcloud_speakers.png'):
    def speakers_long_string(speaker,num):
        string = ""
        for i in range(num):
            string += speaker + ","
        return string

    df = get_score_quotes(quotes, speakers_pageviews).groupby('label')['quotation'].count().to_frame().sort_values(by="quotation")
    df.reset_index(inplace=True)
    d = dict(zip(df.label, df.quotation))
    
    mask = np.array(Image.open("figures/apple_logo_black.png"))
    wordcloud = WordCloud(height=2000, width=1000, mode = "RGBA",
                    background_color = "White", colormap="cividis",repeat=True).generate_from_frequencies(d)

    # plot the WordCloud image                      
    plt.figure(figsize = (8,16), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(path)
    plt.show()   


# ----------------------------------------------------------------- #


def plot_wordcloud(text, path = 'figures/wordcloud.png'): 
    """
    Plot a Word Cloud for representing text data in which the size of each word indicates its frequency or importance.

    Inputs:
        * text (pd.Series): text dataset used for generating the word cloud 
    """
    
    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through text
    for val in text:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()


        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(height=2000, width=1000, mode = "RGBA",
                    background_color = "White",
                    stopwords = stopwords, colormap="copper").generate(comment_words)

    # plot the WordCloud image                      
    plt.figure(figsize = (8,16), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(path)
    plt.show()   


    #plot_wordcloud(' '.join(quotes[quotes.speaker == "steve jobs"].quotation).split(" "))


# ----------------------------------------------------------------- #


def plotly_wordcloud(text):
    """
    Plot the same Word Cloud as 'plot_wordcloud' using plotly this time.

    Inputs:
        * text (pd.Series): text dataset used for generating the word cloud 
    """
    wc = WordCloud(stopwords = set(STOPWORDS),
                   max_words = 200,
                   max_font_size = 100)
    wc.generate(text)
    
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*100)
    new_freq_list
    
    trace = go.Scatter(x=x, 
                       y=y, 
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hoverinfo='text',
                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text',  
                       text=word_list
                      )
    
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
    
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig 


# ----------------------------------------------------------------- #


def split_quote(quote):
    """
    Rewrite the quotation from the quote row with a backline after each 10 words

    Args:
        quote (object): row of the dataframe we want to change 
    Return:
        (object): row of the dataframe with the new format for quotation
    """    
    new_quote = quote.copy()
    quote_cut = str()
    words = quote.quotation.split()
    i=1
    for word in words:
        if (i % 10 == 0):
            quote_cut += '<br>'
        quote_cut += word + ' '
        i += 1
    new_quote.quotation = quote_cut
    return new_quote




# ----------------------------------------------------------------- #


def plot_distrib_val_fame(quotes):
    """
    Plot the distribution of the valence and the fame of the speaker regarding some special days 
        when some events linked to Apple append.

    Inputs:
        quotes (pd.Dataframe): Dataframe of quotes with pageviews colummn already added
    """

    # Create a copy of the dataframe
    quotes_df = quotes.copy()

    # Apply the new format of quotation and change the sentiment by its real score using 'sentiment_score' function
    quotes_df = quotes_df.apply(split_quote, axis = 1)
    quotes_df.sentiment = quotes_df.quotation.progress_apply(sentiment_score)

    # Create new dataframes for each days we choose 
    quotes_FBI = quotes_df[quotes_df.date >= '2016-02-19' ]
    quotes_FBI = quotes_FBI[quotes_FBI.date <= '2016-02-21']

    quotes_iPhone_X = quotes_df[quotes_df.date >= '2017-09-11']
    quotes_iPhone_X = quotes_iPhone_X[quotes_iPhone_X.date <= '2017-09-13']

    quotes_record_Q1 = quotes_df[quotes_df.date >= '2020-01-29'] 
    quotes_record_Q1 = quotes_record_Q1[quotes_record_Q1.date <= '2020-01-31']

    quotes_trillion = quotes_df[quotes_df.date >= '2018-08-01']
    quotes_trillion = quotes_trillion[quotes_trillion.date <= '2018-08-03']

    quotes_designer_leave = quotes_df[quotes_df.date >= '2019-06-26']
    quotes_designer_leave = quotes_designer_leave[quotes_designer_leave.date <= '2019-06-28']

    quotes_event_iPhone7 = quotes_df[quotes_df.date >= '2016-09-07']
    quotes_event_iPhone7 = quotes_event_iPhone7[quotes_event_iPhone7.date <= '2016-09-09']

    # Create the figure by adding the scatter plot of sentiment score against pageviews for each speacial day
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=quotes_FBI.sentiment,
                y=quotes_FBI.pageviews,
                mode = 'markers',
                marker=dict(color=quotes_FBI.sentiment, colorscale='Bluered_r'),
                hovertext= "Speaker : " + quotes_FBI.label + '<br>' + "Quotation : " + quotes_FBI.quotation,
                name="FBI conflict")
        )

    fig.add_trace(
        go.Scatter(x=quotes_iPhone_X.sentiment,
                y=quotes_iPhone_X.pageviews,
                mode = 'markers',
                marker=dict(color=quotes_iPhone_X.sentiment, colorscale='Bluered_r'),
                hovertext= "Speaker : " + quotes_iPhone_X.label + '<br>' + "Quotation : " + quotes_iPhone_X.quotation,
                visible=False,
                name="iPhone X")
        )

    fig.add_trace(
        go.Scatter(x=quotes_record_Q1.sentiment,
                y=quotes_record_Q1.pageviews,
                mode = 'markers',
                marker=dict(color=quotes_record_Q1.sentiment, colorscale='Bluered_r'),
                hovertext= "Speaker : " + quotes_record_Q1.label + '<br>' + "Quotation : " + quotes_record_Q1.quotation,
                visible=False,
                name="Record Q1 earnings")
        )

    fig.add_trace(
        go.Scatter(x=quotes_trillion.sentiment,
                y=quotes_trillion.pageviews,
                mode = 'markers',
                marker=dict(color=quotes_trillion.sentiment, colorscale='Bluered_r'),
                hovertext= "Speaker : " + quotes_trillion.label + '<br>' + "Quotation : " + quotes_trillion.quotation,
                visible=False,
                name="Apple reach 1 trillions")
        )

    fig.add_trace(
        go.Scatter(x=quotes_designer_leave.sentiment,
                y=quotes_designer_leave.pageviews,
                mode = 'markers',
                marker=dict(color=quotes_designer_leave.sentiment, colorscale='Bluered_r'),
                hovertext= "Speaker : " + quotes_designer_leave.label + '<br>' + "Quotation : " + quotes_designer_leave.quotation,
                visible=False,
                name="Jony Ive leaves")
        )

    fig.add_trace(
        go.Scatter(x=quotes_event_iPhone7.sentiment,
                y=quotes_event_iPhone7.pageviews,
                mode = 'markers',
                marker=dict(color=quotes_event_iPhone7.sentiment, colorscale='Bluered_r'),
                hovertext= "Speaker : " + quotes_event_iPhone7.label + '<br>' + "Quotation : " + quotes_event_iPhone7.quotation,
                visible=False,
                name="iPhone 7")
        )

    # Create the buttons for each day
    fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.93,
                    y=-0.25,
                    buttons=list([
                        dict(label="19-21 Feb 2016",
                            method="update",
                            args=[{"visible": [True, False, False, False, False, False]},
                                {"title": "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> FBI–Apple encryption dispute (Feb 2016)"}]),
                        dict(label="07-09 Sep 2016",
                            method="update",
                            args=[{"visible": [False, False, False, False, False, True]},
                                {"title": "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> Event for the presentation of iPhone 7 (Sep 2016)"}]),
                        dict(label="11-13 Sep 2017",
                            method="update",
                            args=[{"visible": [False, True, False, False, False, False]},
                                {"title": "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> Release of the iPhone X (Sep 2017)"}]),
                        dict(label="01-03 Aug 2018",
                            method="update",
                            args=[{"visible": [False, False, False, True, False, False]},
                                {"title": "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> Apple reach $1 trillion in market capitalization (Aug 2018)"}]),
                        dict(label="26-28 Jun 2019",
                            method="update",
                            args=[{"visible": [False, False, False, False, True, False]},
                                {"title": "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> Chief Apple designer, Jony Ive, leaves the company (Jun 2019)"}]),
                        dict(label="29-31 Jan 2020",
                            method="update",
                            args=[{"visible": [False, False, True, False, False, False]},
                                {"title": "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> Record in first quarter results for Apple (Jan 2020)"}])
                    ]),
                )
            ])


    # Set the name for the axis and the title (also set y in log scale)
    fig.update_xaxes(title_text="Sentiment")
    fig.update_yaxes(title_text="Pageviews", type="log")
    fig.update_layout(title_text= "<b>Distribution of quotes according to its valence and the fame of the speaker</b> <br> <br> FBI–Apple encryption dispute (Feb 2016)", title_y=0.95, hoverlabel_align = 'left', xaxis_range=[-1.0, 1.0], template="none")

    fig.show()
    
    # Save the plot in html
    fig.write_html("figures/distribution_valence_fame.html")
