import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        #token_ids = [int(i) for i in txt.strip().split()]

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #tokenizer = None

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)

    return dataloader


#------------------------------------------------------
# IMPLEMENTATION 1: Using Real Data
#------------------------------------------------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257
output_dim = 256
context_length = 1024


token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)


#------------------------------------------------------
# IMPLEMENTATION 2: Using Synthetic / random generated data
# tokenizer = None; token_ids = [int(i) for i in txt.strip().split()]
#------------------------------------------------------
# Generate data from 1-1000
with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):
        f.write(f"{number} ")

# Read data
with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# >> [tensor([[0, 1, 2, 3]]), tensor([[1, 2, 3, 4]])]
next(data_iter)
# >> [tensor([[1, 2, 3, 4]]), tensor([[2, 3, 4, 5]])]

for batch in dataloader:
    pass

last_batch = batch
print(last_batch)
# >> [tensor([[996, 997, 998, 999]]), tensor([[ 997,  998,  999, 1000]])]

# Batched inputs
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# >> Inputs:
 tensor([[992, 993, 994, 995],
        [996, 997, 998, 999]])

# >> Targets:
 tensor([[ 993,  994,  995,  996],
        [ 997,  998,  999, 1000]])


# Data loader with shuffling
torch.manual_seed(123)
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# >> Inputs:
 tensor([[880, 881, 882, 883],
        [112, 113, 114, 115]])

# >> Targets:
 tensor([[881, 882, 883, 884],
        [113, 114, 115, 116]])
