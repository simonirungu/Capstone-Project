
import torch
import numpy as np
import joblib
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.base import BaseEstimator, ClassifierMixin

class BERTsklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, tokenizer, device, max_len=128, threshold=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.threshold = threshold

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, texts):
        if isinstance(texts, str): texts = [texts]
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer.encode_plus(text, truncation=True, max_length=self.max_len, 
                                                   padding='max_length', return_token_type_ids=True)
                ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
                mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)
                token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(self.device)

                outputs = self.model(ids, mask, token_type_ids)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
                all_probs.append(probs)

        return np.array(all_probs)

    def predict_emotions(self, texts):
        preds = self.predict(texts)
        target_cols = ['anger_mapped', 'neutral_mapped', 'joy_mapped', 'surprise_mapped', 
                      'sadness_mapped', 'disgust_mapped', 'fear_mapped']
        return [[target_cols[i] for i, p in enumerate(pred) if p == 1] for pred in preds]

def load_emotion_classifier():
    """Load the emotion classifier for deployment"""
    # Load config
    with open('sklearn_wrapper_config.json', 'r') as f:
        config = json.load(f)

    # Recreate model
    class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.roberta = AutoModel.from_pretrained('roberta-base')
            self.fc = torch.nn.Linear(768, 7)
        def forward(self, ids, mask, token_type_ids):
            _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
            return self.fc(features)

    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTClass()
    model.load_state_dict(torch.load(config['model_weights_path'], map_location='cpu'))
    model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Create wrapper
    classifier = BERTsklearnWrapper(model, tokenizer, device, config['max_len'])
    return classifier

# Global model instance
_emotion_model = None

def get_emotion_model():
    """Get or create emotion model singleton"""
    global _emotion_model
    if _emotion_model is None:
        _emotion_model = load_emotion_classifier()
    return _emotion_model

def predict_emotion(text):
    """Main prediction function for deployment"""
    model = get_emotion_model()
    return model.predict_emotions(text)[0]

def predict_emotion_batch(texts):
    """Batch prediction function"""
    model = get_emotion_model()
    return model.predict_emotions(texts)

def predict_emotion_with_probs(text):
    """Get predictions with probabilities"""
    model = get_emotion_model()
    emotions = model.predict_emotions(text)[0]
    probs = model.predict_proba(text)[0]
    target_cols = ['anger_mapped', 'neutral_mapped', 'joy_mapped', 'surprise_mapped', 
                  'sadness_mapped', 'disgust_mapped', 'fear_mapped']
    return {
        'emotions': emotions,
        'probabilities': dict(zip(target_cols, probs.round(4)))
    }
