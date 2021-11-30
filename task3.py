def task3(quotes):
    
    def sentiment(quote) : 
    vs = analyzer.polarity_scores(quote)['compound']
    if (vs >=0.05) :
        return('positive')
    if (vs <= - 0.05) :
        return('negative') 
    else : return('neutral')  
    
    quotes['sentiment'] = quotes['quotation'].apply(sentiment)  

  return None