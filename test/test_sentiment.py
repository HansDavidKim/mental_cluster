import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### Tester file for sentiment filtering and analysis.
import pandas as pd
from filtering.sentiment_filtering.AbsSentimentFiltering import AbsSentimentFilter
from filtering.sentiment_filtering.sentimentFiltering import VotingSentimentFilter
from filtering.filtering_options import SentimentFilteringOptions

from Dataloader import load_data, preprocess_data, DataFormat

def test_sentiment_filtering():
    # Load data from a sample CSV file
    data = load_data()
    # data = data.head(100)
    
    data = preprocess_data(data, DataFormat.COMMENT_ONLY)  # Limit to 100 rows for testing

    Voting = VotingSentimentFilter()

    # Perform sentiment analysis
    analyzed_data = Voting.sentiment_anlysis(data)
    assert 'prob_vector' in analyzed_data.columns, "Probability vector column not found in analyzed data."
    assert 'prob_logits' in analyzed_data.columns, "Probability logits column not found in analyzed data."

    # Label sentiment
    labeled_data = Voting.label_sentiment(analyzed_data)
    assert 'sentiment' in labeled_data.columns, "Sentiment column not found in labeled data."
    assert 'sentiment_score' in labeled_data.columns, "Sentiment score column not found in labeled data."

    # Save labeled full data
    labeled_data.to_csv('./data/labeled_sentiment_data.csv', index=False)
    print("✅ Labeled data saved to './data/labeled_sentiment_data.csv'.")

    # Filtering the data for a specific sentiment
    sentiment_to_filter = '분노'
    filtered_data = Voting.filter(sentiment_to_filter, labeled_data)
    assert not filtered_data.empty, f"No data found for sentiment: {sentiment_to_filter}"

    # Save filtered subset
    filtered_filename = f'./data/filtered_sentiment_data_{sentiment_to_filter}.csv'
    filtered_data.to_csv(filtered_filename, index=False)
    print(f"✅ Filtered data for sentiment '{sentiment_to_filter}' saved to '{filtered_filename}'.")

    # Preview
    print(f"\nFiltered data preview for sentiment '{sentiment_to_filter}':")
    print(filtered_data[['publishedAt', 'videoTitle', 'text', 'sentiment', 'sentiment_score']].head(10))

if __name__ == "__main__":
    test_sentiment_filtering()
    print("All tests passed successfully!")