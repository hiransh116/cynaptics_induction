import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
#hyperparameters
batch_size=16
block_size=32
temperature=0.9
n_head=16
head_size=16
num_head=4
n_embed=64
max_iters=6400
eval_interval=200
lr=0.002
layers=4
top_k=50


device='cuda' 


#loading the file
url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"
def download_dataset()->None:
    if os.path.exists(DATA_PATH):
        print("File already exits. No changes made.")
        return

    text_file = requests.get(url).text
    with open(DATA_PATH,"w") as f:

        f.write(text_file)


def load_dataset(print_text = False)->str:

    with open(DATA_PATH, "r") as f:
        txt = f.read()

    print("Total Characters in text: ", len(txt))
    if print_text == True:
        print(txt[:10])

    return txt

download_dataset()
txt = load_dataset()
#tokenizer --word based
pattern = r'[\s,.!\n:"?;]|\'s|\'d|--'# cleaned my tokens earlier i just used word based
parts = re.split(pattern,txt) #but the dataset contained many similar punctuation
words=(sorted(list(set(parts))))
vocab=len(words)#vocab_size form 25670 to 14197
wtoi={w:i for i,w in enumerate(words)}
itow={i:w for i,w in enumerate(words)}
encoder=lambda e:[wtoi[token] for token in re.split(pattern,e)]
decoder=lambda d :' '.join([itow[i] for i in d])

df=torch.tensor(encoder(txt),dtype=torch.long)

n=int(len(df)*0.9)
train=df[:n]
test=df[n:]
train,test=train.to(device),test.to(device)



def input_output():
  
  ii = torch.randint(0, len(train) - block_size - 1, (batch_size,))
  input=torch.stack([train[i:i+block_size]for i in ii])
  output=torch.stack([train[i+1:i+1+block_size]for i in ii])
  input,output=input.to(device),output.to(device)
  return input,output

x,y=input_output()




#layers
class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key=nn.Linear(n_embed,head_size,bias=False)
    self.query=nn.Linear(n_embed,head_size,bias=False)
    self.value=nn.Linear(n_embed,head_size,bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout=nn.Dropout(0.4)
    self.head_size=head_size
  def forward(self,x):
    B,T,C=x.shape
    k=self.key(x)#(B,T,C)
    q=self.query(x)#(B,T,C)
    v=self.value(x)#(B,T,C)
    wei=torch.matmul(q,k.transpose(-2,-1))#(B,T,C)*(B,C,T)==(B,T,T)
    wei/=self.head_size**0.5
    tril=torch.tril(torch.ones(T,T))
    wei=wei.masked_fill(tril==0,float('-inf'))
    wei=F.softmax(wei,dim=-1)
    result=torch.matmul(wei,v)#shape-->(B,T,C)
    return result
    



class MultiHeads(nn.Module):
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for number in range(num_heads)])
    self.l1=nn.Linear(64,n_embed)
  def forward(self,x):
    info=torch.cat([hd(x)for hd in self.heads],dim=-1)
    result=self.l1(info)
    return result

class MLP(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.batch=nn.Sequential(
        nn.Linear(n_embd,4*n_embd),
        nn.GELU(),
        nn.Linear(4*n_embd,n_embd),
        nn.Dropout(0.5)
    )

  def forward(self,x):
    return self.batch(x)


class Final(nn.Module):
  def __init__(self,n_embed,num_head):
    super().__init__()
    self.multihead=MultiHeads(num_head,head_size)
    self.mlp=MLP(n_embed)
    self.l1=nn.LayerNorm(n_embed)
  def forward(self,x):
    out=x
    out+=self.multihead(out)#doing residual connections
    out=self.l1(out)
    out+=self.mlp(out)
    return out


import math

class SinCosPositionalEncoding(nn.Module):
  def __init__(self,block_size,n_embed):
    super().__init__()
    matt=torch.zeros(block_size,n_embed)
    matt=matt.to(device)
    positional=torch.arange(block_size).unsqueeze(1).float()#shape-->(t,1)
    divisor=torch.exp( torch.arange(0,n_embed,2).float()*(-math.log(10000.0)/n_embed) )
    positional,divisor=positional.to(device),divisor.to(device)
    divisor=divisor.unsqueeze(0)
    matt[:,0::2]=torch.sin(positional*divisor)
    matt[:,1::2]=torch.cos(positional*divisor)

    self.register_buffer('matt',matt.unsqueeze(0))
   

  def forward(self,x):

    pos_embed=x+self.matt[:,:x.size(1),:]#this additon will be used futher in model class
    return pos_embed
    


class BigramModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.positional_embed=SinCosPositionalEncoding(block_size,n_embed)
    self.token_embed=nn.Embedding(vocab,n_embed)
    self.multi_blocks=nn.Sequential(*[Final(n_embed,num_head) for layer in range(layers)])
    self.l_n=nn.LayerNorm(n_embed)
    
    self.l1=nn.Linear(n_embed,vocab)
  def forward(self,idx,target=None):
    B,T=idx.shape
    token=self.token_embed(idx)
    total_token=self.positional_embed(token)#positonal and token added as per the paper
    out=self.l_n(total_token)
    out=self.l1(out)
    
    if target==None:
      loss=None
    else:
       B,T,C = out.shape
       out=out.view(B*T,C)
       target=target.view(B*T)
       loss=F.cross_entropy(out,target)

    return out,loss

  def generate(self,idx,max_new_tokens):
      for i in range(max_new_tokens):
        idx_cond=idx[:,-block_size:]
        logits,_=self(idx_cond)
        logits=logits[:,-1,:]
        logits[:,0]=float('-inf')# stopping unnecessary generation of space ie [0] to masking it to -inf
        x,order=torch.topk(logits,top_k)
        low_val=x[:,-1].unsqueeze(-1)

        new_logit=torch.where(
          condition=logits<low_val,
          input=torch.tensor(float('-inf')).to(logits.device),
          other=logits,
         
        )

        new_logit/=temperature
        prob=F.softmax(new_logit,dim=-1)
       
        idx_next=torch.multinomial(prob,num_samples=1)
        
        idx=torch.cat((idx,idx_next),dim=1)
      return idx


model =BigramModel()
model=model.to(device)

optimiser=torch.optim.AdamW(model.parameters(),lr=lr)

eval_iters=200
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y=input_output()
            _,loss=model(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out




for iter in range(max_iters):
    if iter%eval_interval==0 or iter==max_iters-1:
        losses=estimate_loss()
        print(f"step {iter}:train loss{losses['train']:.4f},val loss{losses['val']:.4f}")

    # sample a batch of data
    xb,yb=input_output()
    logit,loss=model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# generate from the model

question="what is fate"
context=torch.tensor(encoder(question),dtype=torch.int64).unsqueeze(0).to(device)
print(f'Question is:{question}\n')

ans=(model.generate(context, max_new_tokens=80)[0].tolist())

print(decoder(ans))


  

