
# tokenization for sentiment analysis 
from transformers import DistilBertTokenizerFast

def tokenization(dataset):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return tokenizer(dataset['text'], padding = "max_length", truncation=True, max_length= 512)

