from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from time import time

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model

    tok = AutoTokenizer.from_pretrained("twieland/MIX2_ja-en_helsinki")
    mod = AutoModelForSeq2SeqLM.from_pretrained("twieland/MIX2_ja-en_helsinki")

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        task="translation_ja_to_en",
        model=mod,
        tokenizer=tok,
        device=device
    )

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    input_sentence = model_inputs.get('prompt', None)
    if input_sentence is None:
        return {'message': "No prompt provided"}

    # Run the model
    result = model(
        input_sentence,
        num_beams=2,
        max_length=128,
        truncation=True
    )

    # Return the results as a dictionary
    return result
