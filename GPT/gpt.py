import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.tensorboard as tb
from tqdm import tqdm
from os import path

log_dir = 'logs'

# Hyperparameters
batch_size = 64 # How many independent samples to process at once
block_size = 256 # Context size
epochs = 5000
eval_iters = 500
learning_rate = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.1

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# with open('../data/input.txt') as f:
#     train = f.read()

# with open('../data/more.txt') as f:
#     valid = f.read()

with open('../data/mcs.txt') as f:
    text = f.read()

char_set = sorted(list(set(text)))
vocab_size = len(char_set)

# Encoder
stoi = { ch: i for i, ch in enumerate(char_set) }
encode = lambda s: [stoi[c] for c in s]

# Decoder
itos = { i: ch for i, ch in enumerate(char_set) }
decode = lambda l: ''.join([itos[i] for i in l])

# train_data = torch.tensor(encode(train), dtype=torch.long)
# valid_data = torch.tensor(encode(valid), dtype=torch.long)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]


def make_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    

class MultiHeadAttention(nn.Module):
    """
    Multiple parallel heads of attention.
    """
    class Head(nn.Module):
        """
        A self-attention head.
        """
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embed, head_size, bias=False)
            self.query = nn.Linear(n_embed, head_size, bias=False)
            self.value = nn.Linear(n_embed, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # Input shape: (batch_size, time_step, channels)
            # Output shape: (batch_size, time_step, head_size)
            k = self.key(x)
            q = self.query(x)

            # Compute "affinities"
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:x.shape[1], :x.shape[1]] == 0, float('-inf'))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            # Aggregation
            v = self.value(x)  # (B, T, hs)
            out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MultiHeadAttention.Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):  # TODO: Why do we need this?
    """
    A linear feed-forward network with ReLU activation.
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class GPT(nn.Module):
    """
    A transformer model.
    """

    class Block(nn.Module):
        """
        A transformer block.
        """

        def __init__(self, n_embed, n_head):
            super().__init__()
            head_size = n_embed // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffd = FeedForward(n_embed)
            self.norm1 = nn.LayerNorm(n_embed)
            self.norm2 = nn.LayerNorm(n_embed)
        
        def forward(self, x):
            x = x + self.sa(self.norm1(x))
            x = x + self.ffd(self.norm2(x))
            return x
        
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[GPT.Block(n_embed, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):   # TODO: What is idx? Input indices
        token_emb = self.embed(idx)  # (B, T, C)
        position_emb = self.pos_embed(torch.arange(idx.shape[1], device=device))  # (T, C)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.norm(x)  # (B, T, C)

        logits = self.head(x)  # (B, T, V)

        if targets is not None:  # TODO: what are targets?
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        
        return logits, loss
        
    def generate(self, idx, max_length):
        for _ in range(max_length):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPT()
m = model.to(device)

def train():
    train_logger = tb.SummaryWriter(path.join(log_dir, 'train'), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid'), flush_secs=1)

    global_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in tqdm(range(epochs)):

        # Evaluate the model according to the evaluation interval
        if i % eval_iters == 0 or i == epochs - 1: 
            val_loss = 0
            train_loss = 0
            with torch.no_grad():
                model.eval()
                train_losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    xb, yb = make_batch(train_data)
                    _, loss = model(xb, yb)
                    train_losses[k] = loss.item()
                train_loss = train_losses.mean()

                val_losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    xb, yb = make_batch(valid_data)
                    _, loss = model(xb, yb)
                    val_losses[k] = loss.item()
                
                val_loss = val_losses.mean()

                model.train()
                valid_logger.add_scalar('valid/loss', val_loss.item(), global_step=global_step)
                train_logger.add_scalar('train/loss', train_loss.item(), global_step=global_step)

                print("Epoch: ", i, "Training Loss: ", train_loss.item(), "Validation Loss: ", val_loss.item())

        global_step += 1

        xb, yb = make_batch(train_data)

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)  # TODO: Why set to none?
        loss.backward()
        optimizer.step()
            
    save_model(model)


def save_model(model):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'gpt.pt'))


def load_model():
    r = GPT()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'gpt.pt'), map_location=device))
    return r

# Generate from the GPT model
# Start with an empty context
context = torch.zeros((1, 1), dtype=torch.long, device=device)
if not path.exists(path.join(path.dirname(path.abspath(__file__)), 'gpt.pt')):
    train()
m = load_model().to(device)
print(decode(m.generate(context, 1000)[0].tolist()))