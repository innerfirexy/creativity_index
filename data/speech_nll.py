# %% [markdown]
# ### Process speech data and calculate NLL (Negative Log Likelihood)

# %%
import os
import platform
import pickle
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
print('torch version:', torch.__version__)

# %%
# Load model and tokenizer
# Note: You need to adjust the model path according to your environment
model_name = 'llama3-8b-instruct'
if platform.system() == 'Darwin':
    model_path = f'/Users/xy/models/{model_name}'  # Please modify according to actual situation
else:
    model_path = f'/data2/model/{model_name}'  # Please modify according to actual situation

assert os.path.exists(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Use CPU for computation (if no GPU available)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map='auto', 
                                             trust_remote_code=True).eval()


# %%
# Independent function for computing NLLs
def text_to_nlls(text, tokenizer, model):
    device = model.device
    ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True).to(device)

    # Forward pass
    try:
        output = model(ids)
    except Exception:
        raise
    logits = output.logits.to(device)
    logits = logits.permute(0, 2, 1) # reshape logits from (B, L, V) to (B, V, L)
    shift_logits = logits[:, :, :-1]
    shift_targets = ids[:, 1:]

    # NLL
    loss_fn = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)
    try:
        nlls = loss_fn(log_softmax(shift_logits), shift_targets)
        nlls = nlls.squeeze(0)
    except Exception:
        raise

    return nlls.detach().cpu().numpy()


# %%
def process_speech_data():
    """
    Process all JSON files in the speech folder and calculate NLLs
    """
    folder_path = './speech/'
    assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"
    
    # Get all JSON files in the speech folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in speech folder")
    
    for input_file in json_files:
        full_path = os.path.join(folder_path, input_file)
        print(f"Processing {input_file}...")
        
        # Load data
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text
        all_texts = []
        for item in data:
            text = item.get('text', '')
            if text.strip():  # Only process non-empty text
                all_texts.append(text)
        
        print(f"Found {len(all_texts)} non-empty texts in {input_file}")
        
        # Calculate NLL
        print('Computing NLLs...')
        all_nlls = []
        for text in tqdm(all_texts):
            nlls = text_to_nlls(text, tokenizer, model)
            all_nlls.append(nlls)
        
        # Save intermediate results
        base_name = input_file.split('.')[0]  # Remove .json extension
        pickle_file = f'{base_name}_{model_name}_nlls.pkl'
        pickle.dump(all_nlls, open(pickle_file, 'wb'))
        print(f"Saved intermediate results to {pickle_file}")
        
        # Save text results
        output_text_file = f'{base_name}_{model_name}.txt'
        with open(output_text_file, 'w') as f:
            for nlls in all_nlls:
                f.write(' '.join(f'{nll:.5f}' for nll in nlls) + '\n')
        print(f"Saved text results to {output_text_file}")
        
        # Sanity check
        assert len(all_nlls) == len(all_texts)
        print(f"Sanity check passed for {input_file}")


# %%
# Run the experiment
if __name__ == '__main__':
    print("=" * 50)
    print("Processing SPEECH data...")
    print("=" * 50)
    process_speech_data()
    print("\nSpeech experiments completed!")
