
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline, AutoTokenizer, AutoModel

class TopNClassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10, other_label='Other'):
        self.top_n = top_n
        self.other_label = other_label
        self.top_classes_ = None

    def fit(self, y):
        class_counts = Counter(y)
        self.top_classes_ = [cls for cls, _ in class_counts.most_common(self.top_n)]
        return self

    def transform(self, y):
        return np.where(np.isin(y, self.top_classes_), y, 'Other') 

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class AddressCountryClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_components=100, n_estimators=1000):
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.tnct = TopNClassTransformer()
        self.le = LabelEncoder()
        self.embeddings_model = None
        self.tokenizer = None
        self.pca = PCA(n_components = self.n_components)
        self.rfc = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=7, min_samples_split=8, min_samples_leaf=4, class_weight="balanced_subsample", max_features = "log2", random_state=42)

    def fit(self, X, y):
        y_contained = self.tnct.fit_transform(y)
        y_transformed = self.le.fit_transform(y_contained)

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained('bert-base-uncased')

        embeddings = np.array([self.get_embedding(address) for address in tqdm(X, desc="Extracting embeddings")])
        embeddings_reduced = self.pca.fit_transform(embeddings)

        self.rfc.fit(embeddings_reduced, y_transformed)

        return self

    def predict(self, X):
        embeddings = np.array([self.get_embedding(address) for address in tqdm(X, desc="Extracting embeddings")])
        embeddings_reduced = self.pca.transform(embeddings)
        
        y_pred = self.rfc.predict(embeddings_reduced)
        
        return self.le.inverse_transform(y_pred).tolist()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
