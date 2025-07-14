### Topic Classification Module : KLUE-TC based BERT Model
### Model Name: seongju/klue-tc-bert-base-multilingual-cased

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TopicClassifier:
    '''
    Class for topic classification using a pre-trained model.
    '''
    def __init__(self, model_name: str = 'seongju/klue-tc-bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.mapping = {0: 'IT/과학', 1: '경제', 2: '사회', 3: '생활/문화', 4: '세계', 5: '스포츠', 6: '정치'}

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify the input DataFrame into topic categories.
        
        :param df: DataFrame with a 'cleanTitle' column.
        :return: DataFrame with an additional 'topic' column.
        """
        if 'cleanTitle' not in df.columns:
            raise ValueError("DataFrame must contain a 'videoTitle' column.")

        titles = df['cleanTitle'].tolist()
        inputs = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).tolist()

        df['topic'] = [self.mapping[pred] for pred in predictions]
        return df