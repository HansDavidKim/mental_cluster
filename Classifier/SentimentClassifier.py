import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

id2label = {
    0: "분노", 1: "툴툴대는", 2: "좌절한", 3: "짜증내는", 4: "방어적인", 5: "악의적인",
    6: "안달하는", 7: "구역질 나는", 8: "노여워하는", 9: "성가신", 10: "슬픔", 11: "실망한",
    12: "비통한", 13: "후회되는", 14: "우울한", 15: "마비된", 16: "염세적인", 17: "눈물이 나는",
    18: "낙담한", 19: "환멸을 느끼는", 20: "불안", 21: "두려운", 22: "스트레스 받는", 23: "취약한",
    24: "혼란스러운", 25: "당혹스러운", 26: "회의적인", 27: "걱정스러운", 28: "조심스러운",
    29: "초조한", 30: "상처", 31: "질투하는", 32: "배신당한", 33: "고립된", 34: "충격 받은",
    35: "가난한 불우한", 36: "희생된", 37: "억울한", 38: "괴로워하는", 39: "버려진",
    40: "당황", 41: "고립된(당황한)", 42: "남의 시선을 의식하는", 43: "외로운",
    44: "열등감", 45: "죄책감의", 46: "부끄러운", 47: "혐오스러운", 48: "한심한",
    49: "혼란스러운(당황한)", 50: "기쁨", 51: "감사하는", 52: "신뢰하는", 53: "편안한",
    54: "만족스러운", 55: "흥분", 56: "느긋", 57: "안도", 58: "신이 난", 59: "자신하는"
}
label2id = {v: k for k, v in id2label.items()}

SUB = 'subclass'
VOTING = 'voting'

TITLE_ONLY = 'title'
COMMENT_ONLY = 'comment_only'
TITLE_COMMENT = 'title_comment'

def sub2sup(label: int) -> int:
    return label // 10

class SentimentClassifier:
    def __init__(self, model_name: str = 'hun3359/klue-bert-base-sentiment'):
        """
        Initialize the SentimentClassifier with a pre-trained model and tokenizer.
        Automatically assigns GPU if available.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def classify(self, df: pd.DataFrame, mode: str = SUB, text_format: str = 'title_comment', batch_size: int = 128) -> pd.DataFrame:
        """
        Classify the input DataFrame into sentiment categories with optional batching, tqdm, and GPU support.

        :param df: Input DataFrame containing a column with text.
        :param mode: 'subclass' or 'voting' mode.
        :param text_format: Column name for the text input.
        :param batch_size: Inference batch size.
        :return: DataFrame with added sentiment columns.
        """
        from tqdm import tqdm

        if text_format not in df.columns:
            raise ValueError(f"DataFrame must contain a '{text_format}' column.")

        texts = df[text_format].tolist()
        all_preds = []
        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Classification"):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            if mode == SUB:
                preds = torch.argmax(logits, dim=-1).tolist()
            elif mode == VOTING:
                super_class_prob = torch.zeros((probs.shape[0], 6)).to(self.device)
                for j in range(60):
                    super_class = sub2sup(j)
                    super_class_prob[:, super_class] += probs[:, j]
                probs = super_class_prob
                preds = torch.argmax(probs, dim=-1).tolist()
                preds = [super_class * 10 for super_class in preds]
            else:
                raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'subclass' and 'voting'.")

            all_preds.extend(preds)
            all_probs.extend(probs.cpu().tolist())

        df['sentiment'] = [id2label[p] for p in all_preds]
        df['probabilities'] = all_probs
        df['text_format'] = text_format
        df['mode'] = mode

        return df
