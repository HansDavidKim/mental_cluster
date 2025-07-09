from AbsSentimentFiltering import *
from filtering_options import *

class VotingSentimentFilter(AbsSentimentFilter):
    def __init__(self, 
                 model: str = 'hun3359/klue-bert-base-sentiment', 
                 filtering_option: SentimentFilteringOptions =SentimentFilteringOptions.VOTING
                 ):
        super().__init__(model, filtering_option)

    def label_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Labeling sentiment using voting over superclass groups.
        """
        def get_superclass_and_score(prob_vec):
            # superclass별 확률 합산
            scores = {
                superclass: sum(prob_vec[i] for i in indices)
                for superclass, indices in superclass_to_indices.items()
            }
            top_class = max(scores.items(), key=lambda x: x[1])
            return top_class[0], top_class[1]  # (sentiment, score)

        result = data['prob_vector'].apply(get_superclass_and_score)
        data['sentiment'] = result.map(lambda x: x[0])
        data['sentiment_score'] = result.map(lambda x: x[1])
        return data

    def filter(self, sentiment: str, texts: pd.DataFrame) -> pd.DataFrame:
        """
        Filtering texts based on the sentiment label.
        """
        filtered_texts = texts[texts['sentiment'] == sentiment].copy()
        filtered_texts['sentiment_score'] = filtered_texts['sentiment_score'].round(4)
        return filtered_texts.reset_index(drop=True)