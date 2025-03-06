from langid.langid import LanguageIdentifier, model
import re
import string
import nltk
from nltk.corpus import stopwords



def identify_en(df):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    not_en_idx = set()
    THRESHOLD = 0.95
    for idx, row in df.iterrows():
        score = identifier.classify(row["Comment"])
        if score[0] != "en" or (score[0] == "en" and score[1] <= THRESHOLD):
            not_en_idx.add(idx)
    en_df = df[~df.index.isin(not_en_idx)]
    return en_df


def preprocess_text(text):
    # remove URLs https://www.
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

    # remove HTML Tags: <>
    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    # remove puncs and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text)

    # normalize whitespace
    text = " ".join(text.split())

    # lowercasing
    text = text.lower()

    # Download stopwords if not already downloaded
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = [token.lower() for token in text.split(' ') if token.lower() not in stop_words]
    text = " ".join(text)
    return text


def preprocess(df):
    df["sentence"] = df["sentence"].apply(preprocess_text)
    en_df = identify_en(df)
    return en_df