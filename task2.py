import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.modules import padding
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

device='cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token=tokenizer.eos_token#padding as end of sentence token
model = GPT2LMHeadModel.from_pretrained('gpt2')
model=model.to(device)

model.eval()
vocab_size=tokenizer.vocab_size
print(vocab_size)
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca_prompt(example: dict) -> dict:
    """ format a single alpaca example into the prompt template."""
    if example.get("input") and example["input"].strip():
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


def load_alpaca_dataset(split: str = "train", test_size: float = 0.1, seed: int = 42):
    """
    load the alpaca dataset from HF and apply the prompt template.
    """
    dataset = load_dataset("tatsu-lab/alpaca",split="train",token="HUGGING FACE TOKEN ")
    dataset = dataset.map(format_alpaca_prompt)

    if split == "all":
        return dataset


    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    if split == "train":
        return split_dataset["train"]
    elif split == "test":
        return split_dataset["test"]
    else:
        return split_dataset


if __name__ == "__main__":
    # load and print a few samples
    train_data = load_alpaca_dataset(split="train")
    test_data = load_alpaca_dataset(split="test")

print(train_data[0])
print(train_data[7]["text"])
    


def tokenize_function(examples):
   input=[]
   attention=[]
   label=[]

   for text in examples["text"]:
     stopper="### Response"
     if stopper not in text:
            continue  
     full=tokenizer(
         text,
         truncation=True,
         max_length=256,
         padding=False
     )
     masked_part=text.split(stopper)[0]+"### Response:\n"
     masked_part_token=tokenizer(
         masked_part,
         truncation=True,
         max_length=256,
         padding=False
     )
     masked_part_len=len(masked_part_token["input_ids"])
     labels=torch.tensor(full["input_ids"].copy())
     labels[:masked_part_len]=-100#masking the question part as they will interfere with loss and performance
     input.append(full["input_ids"])
     attention.append(full["attention_mask"])
     label.append(labels)
   return   {
    "input_ids":input,
    "attention_mask":attention,
    "labels":label}


data_collator=DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100  
)
tokenized_train=train_data.map(tokenize_function,   batched=True,   remove_columns=["text", "instruction", "input", "output"])
tokenized_test=test_data.map(tokenize_function,   batched=True,   remove_columns=["text", "instruction", "input", "output"])

train_loader=DataLoader(
    tokenized_train,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator 
)
val_loader=DataLoader(
    tokenized_test,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)
text = '''### Instruction:
Generate a persuasive argument to convince someone to read a book

### Input:
Harry Potter and the Sorcerer's Stone

### Response:
Reading Harry Potter and The Sorcerer's Stone is an enjoyable and enlightening experience. Not only is the story captivating, it also teaches us important lessons about friendship'''
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(
    
    input_ids=inputs["input_ids"],
    
)
logits=outputs.logits
print(inputs)



from transformers import get_scheduler
lr=0.00002
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
epochs=5
training_steps=epochs*len(train_loader)
print(training_steps)
scheduler=get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=training_steps
)



SAVE_DIR      = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

train_loss=[]  
val_loss=[]

for epoch in range(epochs):

  
    model.train()
    total_train_loss=0

    for step, batch in enumerate(train_loader):

        
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["labels"].to(device)

       
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss=outputs.loss

        
        optimizer.zero_grad()   
        loss.backward()         
       
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        optimizer.step()   
        scheduler.step()    

        total_train_loss+=loss.item()

       
        current_lr = optimizer.param_groups[0]['lr']
        if step%50 == 0:
         print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} |LR: {current_lr:.8f}")

    avg_trainloss=total_train_loss/len(train_loader)
    train_loss.append(avg_trainloss)

   
    model.eval()
    total_val_loss = 0

    with torch.no_grad():  
        for batch in val_loader:
            
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels= batch["labels"].to(device)

            outputs = model(
                
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_val_loss+=outputs.loss.item()

    avg_valloss=total_val_loss/len(val_loader)
    val_loss.append(avg_valloss)

    
    print(f"Train Loss:{avg_trainloss:.4f}")
    print(f"Val Loss:{avg_valloss:.4f}\n")
    print(f"Learning_rate:{lr}")

   
    checkpoint_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved→{checkpoint_path}")
fig,ax=plt.subplots(figsize=(12,6))
ax.plot(train_loss,label="training loss")
ax.plot(val_loss,label="validation loss")
ax.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved as loss_curve.png")


#saving checkpoints per epochs
os.makedirs('gpt2_alpaca_checkpoint', exist_ok=True)
model.save_pretrained('gpt2_alpaca_checkpoint')
tokenizer.save_pretrained('gpt2_alpaca_checkpoint')
print("Saved to gpt2_alpaca_checkpoint/")

