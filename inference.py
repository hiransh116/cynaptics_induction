import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( {device})

model     = GPT2LMHeadModel.from_pretrained('gpt2_alpaca_checkpoint')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2_alpaca_checkpoint')
tokenizer.pad_token=tokenizer.eos_token
model=model.to(device)
model.eval()
print("Model loaded!\n")



prompt=(
    ''' Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
' deliver a speech on '
### Input:
'why to Drink alcohol'

'''
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(inputs["input_ids"], max_new_tokens=120, do_sample=True, temperature=0.9,repetition_penalty=1.4,attention_mask=inputs["attention_mask"])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
