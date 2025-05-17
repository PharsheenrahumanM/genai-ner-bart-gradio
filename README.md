## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

### DESIGN STEPS:

### STEP 1: Fine-tune the BART model
Start by fine-tuning the BART model for NER tasks. This involves training the model on a labeled NER dataset with text data that contains named entities (e.g., people, places, organizations).

### STEP 2: Create an API for NER model inference
Develop an API endpoint that takes input text and returns the recognized entities using the fine-tuned BART model.

### STEP 3: Integrate the API with Gradio
Build a Gradio interface that takes user input, passes it to the NER model via the API, and displays the results as highlighted text with identified entities.

### PROGRAM:

# register number :212224230193
#    NAME        :Pharsheen Rahuman M

import os
import re
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

# Call Hugging Face endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data["parameters"] = parameters

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

# Merge tokens for full entities
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

# Find years (like 1900–2100)
def find_years(text):
    matches = re.finditer(r'\b(1[89]\d{2}|20\d{2}|2100)\b', text)
    year_entities = []
    for match in matches:
        year_entities.append({
            "entity": "YEAR",
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "score": 1.0
        })
    return year_entities

# Final NER function
def ner(input):
    output = get_completion(input)
    merged_tokens = merge_tokens(output)
    year_tokens = find_years(input)
    all_entities = merged_tokens + year_tokens
    return {"text": input, "entities": all_entities}

# Gradio Interface
gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER + Year Detection",
    description="Find named entities and years (e.g., 1999, 2023) using the `dslim/bert-base-NER` model and regex.",
    allow_flagging="never",
    examples=[
        "My name is Andrew, I'm building DeeplearningAI and I live in California",
        "Marie Curie won the Nobel Prize in Physics in 1903.",
        "The company was founded in 2020 and expanded in 2023.",
        "My name is Poli, I live in Vienna and work at HuggingFace"
    ]
)

demo.launch(share=True, server_port=int(os.environ['PORT4']))


### OUTPUT:
![image](https://github.com/user-attachments/assets/52a5b3b3-2230-46ea-82e0-0a2c0a4d58c8)

### RESULT:
Thus, The prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model is executed successfully.
