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
def exp_poem_human():
    # Load poetry data
    input_file = './poem/Human_poem.json'
    assert os.path.exists(input_file)
    with open(input_file, 'r', encoding='utf-8') as f:
        poems_data = json.load(f)

    # Extract text
    all_texts = []
    for i, poem in enumerate(poems_data):
        text = poem.get('text', '')
        if text.strip():  # Only process non-empty text
            all_texts.append(text)

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
    output_text_file = f'Human_poem_{model_name}.txt'
    with open(output_text_file, 'w') as f:
        for nlls in all_nlls:
            f.write(' '.join(f'{nll:.5f}' for nll in nlls) + '\n')

# %%
def exp_poem_models():
    input_files = ['./poem/ChatGPT_poem.json',
        './poem/GPT3_poem.json',
        './poem/Llama2-70B-chat_poem.json',
        './poem/Olmo-7B-instruct_poem.json',
        './poem/Tulu2-dpo-70B_poem.json',
    ]
    for input_file in input_files:
        assert os.path.exists(input_file)

    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            poems_data = json.load(f)
        # Extract text 
        all_texts = [poem.get('text', '') for poem in poems_data]

        # Compute NLL
        print(f'computing {input_file}...')
        all_nlls = []
        for text in tqdm(all_texts):
            nlls = text_to_nlls(text, tokenizer, model)
            all_nlls.append(nlls) 

        # Save intermediate
        pickle.dump(all_nlls, open(f'{input_file.split("/")[-1].split(".")[0]}_{model_name}.pkl', 'wb'))

        # Save results
        output_text_file = f'{input_file.split("/")[-1].split(".")[0]}_{model_name}.txt'
        with open(output_text_file, 'w') as f:
            for nlls in all_nlls:
                f.write(' '.join(f'{nll:.5f}' for nll in nlls) + '\n')
        print(f'Saved {output_text_file}')

# %%
# Run the experiment
if __name__ == '__main__':
    # exp_poem_human()
    exp_poem_models()

