from Classifier.SentimentClassifier import *
from Classifier.TopicClassifier import *
from Clustering.Cluster import *
from DataLoader.DataLoad import *

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
    full_parser.add_argument('--topic_filter', type=str, default=config["topic"]["filter"])
    full_parser.add_argument('--sentiment_filter', type=str, default=config["sentiment"]["filter"])
    full_parser.add_argument('--config', type=str, default=args.config)

    return full_parser.parse_args(remaining_args)

if __name__ == "__main__":
    args = parse_args_with_config()

    ### Print arguments using for loop.
    print("\nArguments Received [✔]")
    for key, value in vars(args).items():
        print(f"    {key:<16}: {value}")

    ### Load Data with preprocessing
    print("\nExample data >\n")
    df = load_data('Sentiment_Monitoring_2020-01-01.csv', CSV)
    df = preprocess_video_title(df)

    # print(df.head(4))

    ### Step 1 : Classify Topic
    topic_classifier = TopicClassifier()
    df = topic_classifier.classify(df)

    print(df.head(4))