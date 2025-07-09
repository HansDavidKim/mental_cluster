from AbsSentimentFiltering import *

class VotingSentimentFilter(AbsSentimentFilter):
    def __init__(self, 
                 model: str = 'hun3359/klue-bert-base-sentiment', 
                 filtering_option: SentimentFilteringOptions =SentimentFilteringOptions.VOTING
                 ):
        super().__init__(model, filtering_option)

    def filter(self, texts: pd.DataFrame) -> pd.DataFrame:
        df = self.sentiment_anlysis(texts)
        
        pass
        