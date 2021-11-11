import numpy as np
import matplotlib.pyplot as plt


def bar_plots_quotes(frequency_all, frequency_apple, years):
    barwidth = 0.3

    position_1 = np.arange(len(years)) +  (barwidth / 2)
    position_2 = np.arange(len(years)) + ((3 * barwidth) / 2)

    plt.figure(figsize=[14, 8])

    plt.bar(position_1, frequency_all, width = barwidth, label = 'All quotes', color = [0.9290, 0.6940, 0.1250])
    plt.bar(position_2, frequency_apple, width = barwidth, label = 'Apple quotes', color = [0.6350, 0.0780, 0.1840])

    years_str = [str(i) for i in years]

    plt.xticks([r + barwidth for r in range(len(years))], years_str)
    plt.yscale('log')
    plt.grid(True)
    plt.ylabel('Frequency')
    plt.ylabel('Years')
    plt.legend()

    return 0

def plot_pie_numquote(quote_df, year, head_ = 5):
    ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    ax = quote_df.groupby("speaker")["quotation"].count().sort_values(ascending = False).drop(["none"]).head(head_).plot.pie()
    ax.set(ylabel = None)
    ax.set_title("Repartition of the number of quotes between the %i most commun speakers" %head_ + " for the year %i"  %year )
    plt.show()


def plot_quotes_per_day(quote_df, year):
    quote_df.groupby("date")["quotation"].count().plot()
    plt.title('Numbers of quotes per day in %i' %year)
    plt.ylabel("Number of quotes")
    plt.show()

def plot_numOcc_per_day(quote_df, year):
    quote_df.groupby("date").sum().plot()
    plt.title('Numbers of occurrences of all the quotes per day in %i' %year)
    plt.ylabel("Number of occurrences")
    plt.show()
