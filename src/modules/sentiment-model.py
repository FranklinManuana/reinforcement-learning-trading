from transformers import DistilBertForSequenceClassification

def pretrained_bert(df_train, df_test):
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
