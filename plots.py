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