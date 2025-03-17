from transformers import pipeline


pipeline(model="Khoivudang1209/abte-restaurants-distilbert-base-uncased").save_pretrained("./models/abte-restaurants-distilbert-base-uncased")
pipeline(model="Khoivudang1209/abte-restaurants-sentiment-distilbert").save_pretrained("./models/abte-restaurants-sentiment-distilbert")