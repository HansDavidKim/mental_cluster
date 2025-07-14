from Classifier.SentimentClassifier import *
from Classifier.TopicClassifier import *
from Clustering.Cluster import *
from DataLoader.DataLoad import *
from Visualize import *

import pandas as pd
import numpy as np
import argparse
import tomli

def load_config(path: str) -> dict:
    with open(f"./configs/{path}", "rb") as f:
        return tomli.load(f)

def parse_args_with_config():
    parser = argparse.ArgumentParser(description="Clustering with TOML defaults", add_help=False)
    parser.add_argument('--config', type=str, default="default.toml")
    args, remaining_args = parser.parse_known_args()

    config = load_config(args.config)

    full_parser = argparse.ArgumentParser(description="Full parser with TOML default")
    full_parser = argparse.ArgumentParser(description="Full parser with TOML default")

    full_parser.add_argument('--method', type=str, default=config["cluster"]["method"])
    full_parser.add_argument('--column', type=str, default=config["cluster"]["column"])
    full_parser.add_argument('--topic', type=str, default=config["topic"]["filter"])
    full_parser.add_argument('--sentiment', type=str, default=config["sentiment"]["filter"])
    full_parser.add_argument('--config', type=str, default=args.config)

    return full_parser.parse_args(remaining_args)

if __name__ == "__main__":
    args = parse_args_with_config()

    ### Print arguments using for loop.
    print("\nArguments Received [✔]")
    for key, value in vars(args).items():
        print(f"    {key:<17}: {value}")

    ### Load Data with preprocessing
    df = load_data('Sentiment_Monitoring_2020-01-01.csv', CSV)
    df = preprocess_video_title(df)

    ### Step 1   :   Classify Topic
    topic_classifier = TopicClassifier()
    df = topic_classifier.classify(df)

    ### Step 1.5 :   Filter samples with topic
    if args.topic.lower() != "none":
        df = df[df["topic"] == args.topic]
        print(f"\n[✔] Topic filter applied")
    
    ### Step 2   :   Classify Sentiment
    sentiment_classifier = SentimentClassifier()
    df = sentiment_classifier.classify(df=df, mode='voting', text_format=args.column)

    ### Step 2.5 :   Filter samples with sentiment
    if args.sentiment.lower() != "none":
        df = df[df["sentiment"] == args.sentiment]
        print(f"\n[✔] Sentiment filter applied")

    ### Step 3   :   Embed sentence with SBERT
    cluster = Cluster()
    embedding = cluster.encode(df, args.column)

    ### Step 3.5 :   Cluster embedding vectors with given algorithm
    cluster_label = cluster.cluster(embedding, method=args.method.lower())
    df["cluster"] = list(cluster_label)
    df.to_csv(f"./result{args.column}.csv", index=False)

    print("\nExample data >\n")
    print(df.head(4))

    ### Step 4   :   Reduce dimension
    reduced = reduce_dimension(embedding)

    ### Step 4.5 :   Save figure
    plot_clusters(embedding, cluster_label, f"[{args.column}].png")
