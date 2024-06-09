
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
 
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
 
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)
 
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
 
    # Main training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
 
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous epoch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # Track examples seen instead of tokens
            global_step += 1
 
            # OPTIONAL: Evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
 
        # CALCULATION: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
 
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
 
    return train_losses, val_losses, train_accs, val_accs, examples_seen



def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
 
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1] # Truncate sequences if they are too long
 
    input_ids = input_ids[:min(max_length, supported_context_length)]
 
    input_ids += [pad_token_id] * (max_length - len(input_ids)) # Pad sequences to the longest sequence
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # Add batch dimension
 
    
    with torch.no_grad(): # Model inference without gradient tracking
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
 
    return "spam" if predicted_label == 1 else "not spam"


class CoPE(nn.Module):
    # A novel position encoding method, that allows positions to be conditioned on context.
    # PAPER: Contextual Position Encoding: Learning to Count What’s Important Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar
    def __init__(self, npos_max, head_dim):
        super().__init__ ()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1 , head_dim , npos_max) )
    
    def forward(self , query , attn_logits ):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor

        return logits_ceil * w + logits_floor * (1 - w)


class SelfAttnCope(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.cope = CoPE(npos_max, head_dim)
        self.head_dim = head_dim

    def forward(self, query, key, val, mask):
        # q, k, v have dimensions batch * seq_len * head_dim
        attn_logits = torch.bmm(query, key.transpose(-1, -2))
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_logits += mask.log()
        attn_logits += self.cope(query, attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.bmm(attn, val)

        return out
        

    
        
