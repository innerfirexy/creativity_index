# %% [markdown]
# ### Process poetry data and calculate NLL (Negative Log Likelihood)

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
if platform.system() == 'Darwin':
    model_path = '/Users/xy/models/llama3-8b-base'  # Please modify according to actual situation
else:
    model_path = '/data1/model/llama3-8b-base'  # Please modify according to actual situation

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

    # Forward
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
def exp_poem_human():
    # Load poetry data
    input_file = 'data/poem/Human_poem.json'

    print('Loading poetry data...')
    with open(input_file, 'r', encoding='utf-8') as f:
        poems_data = json.load(f)

    print(f'Total loaded {len(poems_data)} poems')


    # Extract text
    print('Extracting poetry text...')
    all_texts = []
    poem_info = []

    for i, poem in enumerate(poems_data):
        text = poem.get('text', '')
        prompt = poem.get('prompt', '')
        
        if text.strip():  # Only process non-empty text
            all_texts.append(text)
            poem_info.append({
                'index': i,
                'text': text,
                'prompt': prompt,
                'char_count': len(text),
                'word_count': len(text.split())
            })

    print(f'Valid poetry count: {len(all_texts)}')

    # Calculate NLL
    print('tokenizing...')
    all_token_ids = []
    for text in tqdm(all_texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.append(token_ids)

    # Compute
    print('computing...')
    all_nlls = []
    for text in tqdm(all_texts):
        nlls = text_to_nlls(text, tokenizer, model)
        all_nlls.append(nlls)

    # save intermediate
    pickle.dump(all_nlls, open('poem_Human_nlls.pkl', 'wb'))

    # Sanity check
    assert len(all_nlls) == len(all_texts)
    for i, nlls in enumerate(all_nlls):
        assert len(nlls) == len(all_token_ids[i])

    # Save results
    output_text_file = 'poem_Human_llama3-base.txt'
    for i, nlls in enumerate(all_nlls):
        with open(output_text_file, 'a') as f:
            f.write(' '.join(f'{nll:.5f}' for nll in nlls) + '\n')

# %%
# Run the experiment
if __name__ == '__main__':
    exp_poem_human()

