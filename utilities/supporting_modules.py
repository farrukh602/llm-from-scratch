import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tiktoken
# dataloader
class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids=[]
        self.target_ids=[]
        token_ids = tokenizer.encode(text)
        for i in range(0,len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]


# creating dataloader function
def create_dataloader(text, batch_size=64, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
# GPT Architecture
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vocab_size, context_length, emb_dim, num_heads, n_layers, drop_rate, qkv_bias = cfg.values()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(n_layers)]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, in_ids):  # in_ids = input token ids
        batch_size, sequence_len =in_ids.shape
        token_embeds = self.tok_emb(in_ids)
        position_embeds = self.pos_emb(torch.arange(sequence_len, device=in_ids.device))
        x = token_embeds+position_embeds
        x = self.dropout(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)
        x = self.out_head(x)
        
        
        return x



# Multi Head Attention Class
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout,num_heads, qkv_bias=False):
        super().__init__()
        assert d_out%num_heads==0, 'd_out must be divisible by num_heads'
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) #C
        queries = self.W_query(x) #C
        values = self.W_value(x) #C
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)#D
        keys = keys.transpose(1, 2) #E
        queries = queries.transpose(1, 2) #E
        values = values.transpose(1, 2) #E
        attn_scores = queries @ keys.transpose(2, 3) #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) #I
        #J
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #K
        return context_vec

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x+self.shift
        
# GELU Activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
            GELU(),
            nn.Linear(4*cfg['emb_dim'], cfg['emb_dim'])
        )
    def forward(self, x):
        return self.layers(x)


# Transformer Block
 
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vocab_size, context_length, emb_dim, num_heads, n_layers, drop_rate, qkv_bias = cfg.values()
        embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.layernorm1 = LayerNorm(emb_dim)
        self.layernorm2 = LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(d_in=emb_dim, d_out=emb_dim, context_length=context_length, dropout=drop_rate,
                           num_heads=num_heads, qkv_bias=qkv_bias )
        self.dropout = nn.Dropout(p=drop_rate)
        self.ffn = FeedForward(cfg)
    def forward(self, x):
        #A
        skip_connection=x
        x = self.layernorm1(x) # pre normalization (before multihead atention and feed forward)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + skip_connection

        #B
        skip_connection=x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x+skip_connection
        
        return x
        
        
# Utility functions for calculating losses

## Cross Entropy Loss
def cross_entropy_batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def loss_calculator(data_loader, model, loss_function, device, num_batches=None):
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader) # A
    else:
        num_batches = min(num_batches, len(data_loader)) # B
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_function(input_batch, target_batch, model, device)
            total_loss += loss
            break
        else:
            break
    return total_loss / num_batches
        