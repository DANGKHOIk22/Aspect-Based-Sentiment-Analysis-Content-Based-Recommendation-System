from fastapi import HTTPException, status
from transformers import pipeline

async def predict_label(input_data):
    token_classifier = pipeline(
        model="Khoivudang1209/abte-restaurants-distilbert-base-uncased",
        aggregation_strategy="simple"
    )

    classifier = pipeline(
        model="Khoivudang1209/abte-restaurants-sentiment-distilbert"
    )

    try:
        text_input = input_data.text
        results = token_classifier(text_input)
        sentence_tags = " ".join([result['word'] for result in results])
        pred_label = classifier(f'{text_input} [SEP] {sentence_tags}')
        
        return {
            'sentence_tags': sentence_tags,
            'label': pred_label[0]["label"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing sentiment classification: {str(e)}"
        )