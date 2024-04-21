import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import LanguageModel, beam_search, SpeechDataset, one_hot, vocab_size

from os import path
import torch.utils.tensorboard as tb
from tqdm import tqdm
from os import path

log_dir = 'logs'

# Hyperparameters
batch_size = 128
epochs = 500
learning_rate = 3e-1

# Model parameters
num_blocks = 5
num_channels = 50
kernel_size = 2

# Beam search parameters
beam_size = 20
n_results = 10
beam_length = 100
average_log_likelihood = True

device = 'mps' if torch.backends.mps.is_available() else 'cpu'


class TCN(nn.Module, LanguageModel):
    """
    A Temporal Convolutional Network for sequence generation.
    """
    class CausalConv1dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            """
            A casual convolution block with a ReLU activation function.
            """
            super().__init__()

            padding = (kernel_size - 1) * dilation
            self.model = nn.Sequential(
                nn.ConstantPad1d((padding, 0), 0),
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size,
                          stride=1,
                          dilation=dilation),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.ConstantPad1d((padding, 0), 0),
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size,
                          stride=1,
                          dilation=dilation),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            if in_channels == out_channels:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Conv1d(in_channels, out_channels, 1)

        def forward(self, x):
            return self.model(x) + self.residual(x)

    def __init__(self):
        super().__init__()
        self.first_char_prob = nn.Parameter(torch.randn(vocab_size))
        layers = []
        for i in range(num_blocks):
            in_channels = vocab_size if i == 0 else num_channels
            dilation = 2**i
            layers.append(TCN.CausalConv1dBlock(in_channels,
                                                num_channels,
                                                kernel_size,
                                                dilation))
        self.tcn_blocks = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_channels, vocab_size, 1)
    
    def forward(self, x):
        """
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        if torch.numel(x) == 0:
            fcp = self.first_char_prob[np.newaxis, :, np.newaxis]
            expanded_first_char_prob = fcp.expand((x.shape[0], -1, -1))
            return expanded_first_char_prob
    
        x = self.tcn_blocks(x)
        x = self.final_conv(x)

        fcp = self.first_char_prob[np.newaxis, :, np.newaxis]
        expanded_first_char_prob = fcp.expand((x.shape[0], -1, -1))
        return torch.cat((expanded_first_char_prob, x), dim=2)

    def predict_all(self, some_text):
        """
        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        one_hot_enc = one_hot(some_text).unsqueeze(0)
        logits = self.forward(one_hot_enc)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs.squeeze(0)

    def generate(self, max_length=20):
        """
        Generate a string using the beam search.
        """

        generated = []
        for _ in range(max_length):
            generated.append(
                beam_search(self,
                            beam_size,
                            n_results,
                            max_length,
                            average_log_likelihood))
            print(generated[-1])


def save_model(model):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.pt'))


def load_model():
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.pt'), map_location='cpu'))
    return r


def train():
    model = TCN().to(device)

    data_loader = torch.utils.data.DataLoader(
        SpeechDataset(dataset_path='../data/mcs.txt', transform=one_hot),
        batch_size=batch_size,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        SpeechDataset(dataset_path='../data/mcs_more.txt', transform=one_hot),
        batch_size=batch_size,
        shuffle=True)
    
    train_logger = tb.SummaryWriter(path.join(log_dir, 'train'), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid'), flush_secs=1)

    global_step = 0
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    valid_accs = []

    for i in tqdm(range(epochs)):
        model.train()
        for batch in data_loader:
            batch = batch.to('mps')
            optimizer.zero_grad()
            logits = model(batch[:, :, :-1])
            loss_val = loss(logits, batch.argmax(dim=1))
            acc = (logits.argmax(dim=1) == batch.argmax(dim=1)).float().mean()

            train_logger.add_scalar('train/loss', loss_val.item(), global_step=global_step)
            train_logger.add_scalar('train/acc', acc.item(), global_step=global_step)

            global_step += 1

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
        model.eval()

        for batch in valid_loader:
            batch = batch.to('mps')
            logits = model(batch[:, :, :-1])
            acc = (logits.argmax(dim=1) == batch.argmax(dim=1)).float().mean()
            valid_accs.append(acc.item())

        scheduler.step()

        if (i + 1) % 5 == 0:
            print("Epoch: ", i, "Validation Accuracy: ", torch.tensor(valid_accs).mean().item())
        valid_logger.add_scalar('valid/acc', torch.tensor(valid_accs).mean(), global_step=global_step)
    
    save_model(model)


# Generate from the TCN model
if path.exists(path.join(path.dirname(path.abspath(__file__)), 'tcn.pt')):
    load_model().generate()
else:
    train()
    load_model().generate()
