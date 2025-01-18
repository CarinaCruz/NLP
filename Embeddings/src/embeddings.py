# Databricks notebook source
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class BertEncoding():
    """
    Class to encode text using BERT.
    """
    
    def __init__(self, device, model_path="google-bert/bert-base-multilingual-cased"):
        """
        Initializes the BERT tokenizer and model.
        
        :param model_path: Path to the pre-trained BERT model.
        :param cuda: type of processing architecture
        """    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True)
        self.model = AutoModel.from_pretrained(model_path, force_download=True)
        self.device = device
    
    def encode(self, text):
        """
        Encodes the text into embeddings using BERT.
        
        :param text: Text to be encoded.
        :return: Embeddings of the text.
        """                
        self.model.to(self.device)
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.pooler_output
        return np.array(embeddings).flatten()


class ModernBertEncoding():
    """
    Class to encode text using BERT.
    """
    
    def __init__(self, device, model_path="answerdotai/ModernBERT-base"):
        """
        Initializes the BERT tokenizer and model.
        
        :param model_path: Path to the pre-trained BERT model.
        :param cuda: type of processing architecture
        """    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True)
        self.model = AutoModel.from_pretrained(model_path, force_download=True)
        self.device = cuda
    
    def encode(self, text):
        """
        Encodes the text into embeddings using BERT.
        
        :param text: Text to be encoded.
        :return: Embeddings of the text.
        """        
        self.model.to(self.device)
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return np.array(embeddings).flatten()