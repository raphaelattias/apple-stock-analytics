import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

import plotly.graph_objs as go
alpha_ = 0.7
figsize = [10.0, 6.0]
DPI = 250

def bar_plots_quotes(frequency_all, frequency_apple, years):
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

def plot_pie_numquote(quote_df, year, head_ = 5):
    ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    ax = quote_df.groupby("speaker")["quotation"].count().sort_values(ascending = False).drop(["none"]).head(head_).plot.pie()
    ax.set(ylabel = None)
    ax.set_title("Repartition of the number of quotes between the %i most commun speakers" %head_ + " for the year %i"  %year )

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.show()


def plot_quotes_per_day(quote_df, year):
    quote_df.groupby("date")["quotation"].count().plot(alpha = alpha_)
    plt.title('Numbers of quotes per day in %i' %year)
    plt.ylabel("Number of quotes")

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.show()

def plot_numOcc_per_day(quote_df, year):
    quote_df.groupby("date").sum().plot(alpha = alpha_)
    plt.title('Numbers of occurrences of all the quotes per day in %i' %year)
    plt.ylabel("Number of occurrences")

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = DPI

    plt.show()

    
def plot_wordcloud(text): 
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

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "
    
    mask = np.array(Image.open("figures/apple_logo_black.png"))
    wordcloud = WordCloud(height=4000, width=1000, mode = "RGBA",
                    background_color = "White",
                    stopwords = stopwords, colormap="copper").generate(comment_words)

    # plot the WordCloud image                      
    plt.figure(figsize = (32, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('figures/wordcloud.png')
    plt.show()   


    #plot_wordcloud(' '.join(quotes[quotes.speaker == "steve jobs"].quotation).split(" "))


def plotly_wordcloud(text):
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