### Tester file for sentiment filtering and analysis.
import pandas as pd
from filtering.sentiment_filtering.sentimentFiltering import VotingSentimentFilter
from filtering.filtering_options import SentimentFilteringOptions

def test_sentiment_filtering():
    # Load dafa from a sample CSV file
    data = pd.read_csv('./data/Sentiment_Monitoring_2020-01-01.csv')
    data = data.head(100)  # Limit to 100 rows for testing

    Voting = VotingSentimentFilter()
    # Perform sentiment analysis
    analyzed_data = Voting.sentiment_anlysis(data)
    assert 'prob_vector' in analyzed_data.columns, "Probability vector column not found in analyzed data."
    assert 'prob_logits' in analyzed_data.columns, "Probability logits column not found in analyzed data."

    # Label sentiment
    labeled_data = Voting.label_sentiment(analyzed_data)
    assert 'sentiment' in labeled_data.columns, "Sentiment column not found in labeled data."
    assert 'sentiment_score' in labeled_data.columns, "Sentiment score column not found in labeled data."

    # Saving the labeled data to a CSV file
    labeled_data.to_csv('./data/labeled_sentiment_data.csv', index=False)
    print("Sentiment filtering and labeling completed successfully. Labeled data saved to 'labeled_sentiment_data.csv'.")

    # Filtering the data for a specific sentiment
    sentiment_to_filter = '분노'  # Example sentiment
    filtered_data = Voting.filter(sentiment_to_filter, labeled_data)
    assert not filtered_data.empty, f"No data found for sentiment: {sentiment_to_filter}"
    print(f"Filtered data for sentiment '{sentiment_to_filter}':")
    print(filtered_data[['publishedAt', 'videoTitle', 'text', 'sentiment', 'sentiment_score']].head(10))