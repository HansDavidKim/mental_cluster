import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualize_reduced_logits import visualize_reduced_logits

import pandas as pd
import cluster.Cluster as Cluster
from cluster.DimReductionOption import DimReductionOptions

if __name__ == '__main__':
    data_path = './data/labeled_sentiment_data.csv'
    n_components = 2
    perplexity = 30
    cluster_object = Cluster.Cluster(n_components=15)
    data = pd.read_csv(data_path)
    data = cluster_object.fit_transform(data)
    visualize_reduced_logits.visualize_reduced_logits(data_path, n_components, perplexity)