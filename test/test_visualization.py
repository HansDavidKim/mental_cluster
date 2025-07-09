import visualize_reduced_logits
import pandas as pd

if __name__ == '__main__':
    data_path = './data/labeled_sentiment_data.csv'
    n_components = 2
    perplexity = 30
    visualize_reduced_logits.visualize_reduced_logits(data_path, n_components, perplexity)