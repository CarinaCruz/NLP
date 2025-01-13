from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class BertEncoding():
    """
    Class to encode text using BERT.
    """
    
    def __init__(self, model_path="google-bert/bert-base-multilingual-cased"):
        """
        Initializes the BERT tokenizer and model.
        
        :param model_path: Path to the pre-trained BERT model.
        """    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True)
        self.model = AutoModel.from_pretrained(model_path, force_download=True)
    
    def encode(self, text):
        """
        Encodes the text into embeddings using BERT.
        
        :param text: Text to be encoded.
        :return: Embeddings of the text.
        """        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.pooler_output
        return np.array(embeddings).flatten()    
