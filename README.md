# Influence of the Media on Apple Stock Market

### Abstract

Apple has been a market leader in the world of technology ever since the launch of its first product. Furthermore, media are more and more being used to study their impact on stock market movements. In this project, we aim to show that the rises and falls in stock prices of Apple correlate to the extent that people are talking about Apple in the media, the fame of the speakers and the way they are talking about the company. Firstly, we will wonder what is the role of the media coverage in explaining stock market fluctuations ? Then, we will examine who are the individuals who have influence over potential customers, and do these influencers have an impact on the company image and eventually, on the stock market ? Eventually, we will ask what is the influence of the people's opinions about Apple expressed in the media on the stock market ?

***
### Data sets used : 
* `Quotebank DataSet` : To investigate the reciprocal relationships between the fluctuations of the Apple stock market and media coverage related to this company, we study mentions of Apple using time series analysis. We filter out any quotes that are not related to the company, the products or its direction board. The filtered dataset is then saved as a pickle file containing approximately 300.000 quotes out of the original 234 millions. This new dataset provides a great preview of Apple mentions in the media. We will then analyse to what extent Apple is mentioned according to time in the media, who are the speakers talking about this company and the way they are talking about it.
* `Wikipedia API` : We use the Wikipedia API to recover informations about the speakers. We will classify them as having a significant influence or a low influence over potential customers based on the Wikipedia page view statistics per year. The idea is to get all the Wiki qids speakers that have have spoken in our filtered quotes dataset. Then, with the Wikipedia API, we would get all the needed informations about the speakers.
* `Yahoo Finance API` : Eventually, we use the Yahoo Finance API to recover informations about the stock markets. This API provides quick and easy access to finance metrics for any stock or index. Among the many financial metrics available we decided to focus on the daily stock price and volume. The former will be an indicator of the long term health of the stock, and the latter of the daily volatility it may experience. In this milestone we compare the Apple stock ($AAPL) to the S&P500 ($SPY) from 2008 until 2020, which is is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the US. We have chosen this equity index as it is one of the most tracked indices and generally an indicator of the overall health of the US stock market.

***
### Methods : 
* **Method to classify the fame of people annually :** For the subtask 2, we classify people as having a significant influence or a low influence over potential customers based on the Wikipedia page view statistics per year. We set the threshold at 10 000 page views per year, i.e. if people have more than 10 000 page views on their Wikipedia page one specific year, they will be considered as having a significant influence over customers on that year. For this specific subtask - and only for this one - we won’t consider the quotes for which the speakers are non identified (classified as ‘none’).   
* **Methods to determine the author's opinion from a piece of text :** For the subtask 3, we apply sentiment analysis to the quotebank dataset to judge the opinion and the type of sentiments expressed in the quotes. The quotes are classified into three categories : positive, negative and neutral. To do so, we will use a machine learning model pre-trained (https://github.com/cardiffnlp/tweeteval) for sentiment analysis called [roBERTa](https://arxiv.org/pdf/2010.12421.pdf). This model has been trained on approximately 58 millions tweets to predict their sentiments, and reached the best results out of a wide range of complex models. We then analyze the correlation between stock market movements of Apple company and sentiments expressed in the quotes. 


***
### Proposed timeline : 
**Week 8 :** *(12/11 : P2 due)*

**Week 9 :** Begin subtasks 1 and 2 (What is the role of the media coverage in explaining stock market fluctuations ? Who are the individuals who have influence over potential customers, and do these influencers have an impact on the Apple company image and eventually, on the stock market ?)

**Week 10 :** Finish subtasks 1 and 2.

**Week 11 :** Subtask 3 (What is the influence of the public opinions or emotions about Apple expressed in the media on the stock market ?) 

**Week 12 :** Create a website with interactive graphs for our data story. Write the README. 

**Week 13 :** Finalize the website, the notebook, the code architecture and the README. *(17/12 : P3 due)*

***
### Organization within the team : 
* [@Bapitou](https://github.com/Bapitou) : Quotebank EDA and filtering (Week 8 P2), task 1 (Week 9-10), task 3 (Week 11), finalize the website (Week 12-13)
* [@camillefrayssinhes](https://github.com/camillefrayssinhes): Quotebank EDA and filtering (Week 8 P2), task 2 (Week 9-10), Writing up the README and the datastory for the website (Week 11-12-13)
* [@gaspardvilla](https://github.com/gaspardvilla): Wikipedia API recover the annual number of page views for each speaker (Week 8 M2, Week 9), task 2 (Week 9-10), task 3  (Week 11), finalize the website (Week 12-13)
* [@raphaelattias](https://github.com/raphaelattias) : Yahoo Finance API EDA and filtering (Week 8 P2), task 1 (Week 9-10), create the website (Week 11-12)

*** 
### Code Architecture
#### Quotebank 
`dataloader.py` : loading and filtering the quotebank dataset according to our needs 

`quotebankexploration.py` : performing EDA of the quotebank dataset before and after filtering 
#### Yahoo Finance 
`finance.py` : performing EDA of the Yahoo Finance dataset 
#### Wikipedia
`wikipedia.py` : pre-processing of the wikidata 

`query_wikidata_dump.py` : recovering additional metadata about the speakers in the Quotebank dataset (identified with their QIDs) 
#### Plots
`plots.py` : visualizing the graphs 
#### Jupyter Notebook
`main.ipynb` : EDA and filtering of the datasets we use, subtasks analyses 






