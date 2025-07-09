from abc import ABC, abstractmethod
from filtering_options import SentimentFilteringOptions
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

import torch
from torch.nn.functional import softmax
from tqdm import tqdm 

class AbsSentimentFilter(ABC):
    '''
    Abstract class for sentiment filtering.
    '''
    def __init__(self, 
                 model: str = 'hun3359/klue-bert-base-sentiment', 
                 filtering_option: SentimentFilteringOptions = SentimentFilteringOptions.VOTING
                 ):
        ### Loading huggingface model for sentiment analysis.
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.eval()
        self.filtering_option = filtering_option

    def sentiment_anlysis(
            self, 
            texts: pd.DataFrame, 
            batch_size: int = 32
            ) -> pd.DataFrame:
        '''
        Method for performing sentiment analysis on the input texts.
        :param texts: List of texts to analyze.
        :return: DataFrame with sentiment scores.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        sequences = texts["sequence"].tolist()
        all_probs = []
        all_logits = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="Analyzing sentiments"):
            batch = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = softmax(logits, dim=-1).cpu().numpy()  # (batch_size, num_classes)
                all_probs.extend(probs)
                all_logits.extend(logits.cpu().numpy())

        ### Logits are required for clustering other than probability vectors.
        texts['prob_logits'] = all_logits
        texts['prob_vector'] = all_probs
        return texts
    
    @abstractmethod
    def label_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Abstract method for labeling sentiment.
        :param data: DataFrame with sentiment scores.
        :return: DataFrame with labeled sentiments.
        '''
        pass

    @abstractmethod
    def filter(self, sentiment: str, texts: pd.DataFrame) -> pd.DataFrame:
        '''
        Abstract method for filtering sentiment.
        :param texts: List of texts to filter.
        :return: List of filtered sentiments.
        '''
        pass