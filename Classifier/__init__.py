from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TopicClassifier:
    '''
    Class for topic classification using a pre-trained model.
    '''
    def __init__(self, model_name: str = 'seongju/klue-tc-bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def classify(self, texts: list) -> list:
        '''
        Classify the input texts into topics.
        :param texts: List of texts to classify.
        :return: List of predicted topic labels.
        '''
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).tolist()
        
        return predictions

topic_mapping = {
    0: 'IT/과학', 
    1: '경제', 
    2: '사회', 
    3: '생활문화', 
    4: '세계', 
    5: '스포츠', 
    6: '정치'
}