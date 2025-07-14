### Sentiment Classifier Module : DataLoader for Sentiment Analysis
### Model : hun3359/klue-bert-base-sentiment

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

id2label = {
    0: "분노",
    1: "툴툴대는",
    2: "좌절한",
    3: "짜증내는",
    4: "방어적인",
    5: "악의적인",
    6: "안달하는",
    7: "구역질 나는",
    8: "노여워하는",
    9: "성가신",
    10: "슬픔",
    11: "실망한",
    12: "비통한",
    13: "후회되는",
    14: "우울한",
    15: "마비된",
    16: "염세적인",
    17: "눈물이 나는",
    18: "낙담한",
    19: "환멸을 느끼는",
    20: "불안",
    21: "두려운",
    22: "스트레스 받는",
    23: "취약한",
    24: "혼란스러운",
    25: "당혹스러운",
    26: "회의적인",
    27: "걱정스러운",
    28: "조심스러운",
    29: "초조한",
    30: "상처",
    31: "질투하는",
    32: "배신당한",
    33: "고립된",
    34: "충격 받은",
    35: "가난한 불우한",
    36: "희생된",
    37: "억울한",
    38: "괴로워하는",
    39: "버려진",
    40: "당황",
    41: "고립된(당황한)",
    42: "남의 시선을 의식하는",
    43: "외로운",
    44: "열등감",
    45: "죄책감의",
    46: "부끄러운",
    47: "혐오스러운",
    48: "한심한",
    49: "혼란스러운(당황한)",
    50: "기쁨",
    51: "감사하는",
    52: "신뢰하는",
    53: "편안한",
    54: "만족스러운",
    55: "흥분",
    56: "느긋",
    57: "안도",
    58: "신이 난",
    59: "자신하는"
},
label2id = {
    "분노": 0,
    "툴툴대는": 1,
    "좌절한": 2,
    "짜증내는": 3,
    "방어적인": 4,
    "악의적인": 5,
    "안달하는": 6,
    "구역질 나는": 7,
    "노여워하는": 8,
    "성가신": 9,
    "슬픔": 10,
    "실망한": 11,
    "비통한": 12,
    "후회되는": 13,
    "우울한": 14,
    "마비된": 15,
    "염세적인": 16,
    "눈물이 나는": 17,
    "낙담한": 18,
    "환멸을 느끼는": 19,
    "불안": 20,
    "두려운": 21,
    "스트레스 받는": 22,
    "취약한": 23,
    "혼란스러운": 24,
    "당혹스러운": 25,
    "회의적인": 26,
    "걱정스러운": 27,
    "조심스러운": 28,
    "초조한": 29,
    "상처": 30,
    "질투하는": 31,
    "배신당한": 32,
    "고립된": 33,
    "충격 받은": 34,
    "가난한 불우한": 35,
    "희생된": 36,
    "억울한": 37,
    "괴로워하는": 38,
    "버려진": 39,
    "당황": 40,
    "고립된(당황한)": 41,
    "남의 시선을 의식하는": 42,
    "외로운": 43,
    "열등감": 44,
    "죄책감의": 45,
    "부끄러운": 46,
    "혐오스러운": 47,
    "한심한": 48,
    "혼란스러운(당황한)": 49,
    "기쁨": 50,
    "감사하는": 51,
    "신뢰하는": 52,
    "편안한": 53,
    "만족스러운": 54,
    "흥분": 55,
    "느긋": 56,
    "안도": 57,
    "신이 난": 58,
    "자신하는": 59
}

SUB = 'subclass'
VOTING = 'voting'

TITLE_ONLY = 'title_only'
COMMENT_ONLY = 'comment_only'
TITLE_COMMENT = 'title_comment'

### Sub-class to super-class in sentiment classification
def sub2sup(label: int) -> int:
    return label // 10

class SentimentClassifier:
    def __init__(self, model_name: str = 'hun3359/klue-bert-base-sentiment'):
        """
        Initialize the SentimentClassifier with a pre-trained model and tokenizer.
        
        :param model_name: Name of the pre-trained model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def classify(self, df: pd.DataFrame, mode: str = SUB, text_format: str = 'title_comment') -> pd.DataFrame:
        """
        Classify the input DataFrame into sentiment categories.
        
        :param df: DataFrame with a 'text' column.
        :param mode: Classification mode, either 'subclass' or 'voting'.
        :return: DataFrame with an additional 'sentiment' column.
        """
        if text_format not in df.columns:
            raise ValueError(f"DataFrame must contain a '{text_format}' column.")

        texts = df['text'].tolist()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        if mode == SUB:
            predictions = torch.argmax(logits, dim=-1).tolist()
            df['sentiment'] = [id2label[pred] for pred in predictions]
        elif mode == VOTING:
            # For voting, we sum the probabilities of super-classes
            super_class_prob = torch.zeros((probabilities.shape[0], 6))  # 6 super-classes

            for i in range(60):  # 60 sub-classes
                super_class = sub2sup(i)
                super_class_prob[:, super_class] += probabilities[:, i]
            predictions = torch.argmax(super_class_prob, dim=-1).tolist()
            probabilities = super_class_prob
            df['sentiment'] = [id2label[pred * 10] for pred in predictions]  # Map to super-class labels
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'subclass' and 'voting'.")
        
        df['probabilities'] = probabilities.tolist()
        df['text_format'] = text_format
        df['mode'] = mode
        return df