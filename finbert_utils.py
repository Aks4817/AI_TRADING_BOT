from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
import time
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":

    tensor, sentiment = estimate_sentiment([
"Global markets reel from unexpected downturn, sending shockwaves through investors. Economic instability and geopolitical tensions exacerbate fears, with stocks plunging and analysts predicting prolonged turbulence in the financial landscape."])
    # start_time = time.time()
    print(tensor, sentiment)
    # end_time = time.time()

    elapsed_time = end_time - start_time

    # print("\nElapsed time:", elapsed_time, "seconds")

    # print(torch.cuda.is_available())