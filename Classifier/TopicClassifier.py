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

    def classify(self, df: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
        """
        Classify the input DataFrame into topic categories with tqdm and batch processing.

        :param df: DataFrame with a 'title' column.
        :param batch_size: Number of samples per batch.
        :return: DataFrame with an additional 'topic' column.
        """
        if 'title' not in df.columns:
            raise ValueError("DataFrame must contain a 'title' column.")

        from tqdm import tqdm  # 함수 내부에서 import

        titles = df['title'].tolist()
        all_preds = []

        for i in tqdm(range(0, len(titles), batch_size), desc="Topic Classification"):
            batch_titles = titles[i:i + batch_size]
            inputs = self.tokenizer(batch_titles, return_tensors='pt', padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).tolist()

            all_preds.extend(preds)

        df['topic'] = [self.mapping[p] for p in all_preds]
        return df
