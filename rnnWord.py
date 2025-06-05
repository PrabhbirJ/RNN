import pandas as pd
import re
import random
import numpy as np
df = pd.read_csv("PoetryFoundationData.csv")

def clean_text(text):
    text = text.lower()
    # Keep letters, space, and select punctuation
    text = re.sub(r"[^a-z\s.,!?;:'\"-]", '', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned_title'] = df['Title'].apply(clean_text)

# CHANGE 1: Word-level tokenization instead of character-level
def tokenize_titles(titles):
    """Convert titles to word tokens"""
    word_vocab = set()
    tokenized = []
    
    for title in titles:
        # Split into words (handles punctuation attached to words)
        words = title.split()
        # Filter out empty strings
        words = [w for w in words if w.strip()]
        tokenized.append(words)
        word_vocab.update(words)
    
    return tokenized, sorted(list(word_vocab))

# CHANGE 2: Process titles as word sequences
titles = df['cleaned_title'].tolist()
START_TOKEN = '<START>'
END_TOKEN = '<END>'

# Tokenize all titles
tokenized_titles, word_vocab = tokenize_titles(titles)

# Add start/end tokens to vocabulary
word_vocab = [START_TOKEN, END_TOKEN] + word_vocab
vocab_size = len(word_vocab)

# Create word-to-index mappings
word_to_idx = {word: i for i, word in enumerate(word_vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# CHANGE 3: Add start/end tokens to each title sequence
processed_titles = [[START_TOKEN] + title + [END_TOKEN] for title in tokenized_titles]

# CHANGE 4: Encode titles as word indices instead of character indices
def encode_text_words(word_sequence):
    """Convert word sequence to indices"""
    return [word_to_idx[word] for word in word_sequence]

encoded_titles = [encode_text_words(title) for title in processed_titles]

# CHANGE 5: Update sequence creation (same logic, but now with words)
def create_sequences(data_list):
    inputs = []
    targets = []
    for seq in data_list:
        if len(seq) < 2:
            continue
        inputs.append(seq[:-1])  # all except last word
        targets.append(seq[1:])  # all except first word
    return inputs, targets

X, Y = create_sequences(encoded_titles)
print(f"Number of sequences: {len(X)}")
print(f"Vocabulary size: {vocab_size}")
print(f"Sample words: {word_vocab[:10]}")

# CHANGE 6: Update batch generator (same logic, different pad token)
def batch_generator(X, Y, batch_size, pad_token=0):  # 0 = <START> token as padding
    n = len(X)
    indices = list(range(n))
    while True:
        random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = [X[i] for i in batch_idx]
            y_batch = [Y[i] for i in batch_idx]
            
            # Find max length in this batch for padding
            max_len = max(len(seq) for seq in x_batch)
            
            # Pad sequences with pad_token
            x_batch_padded = [seq + [pad_token] * (max_len - len(seq)) for seq in x_batch]
            y_batch_padded = [seq + [pad_token] * (max_len - len(seq)) for seq in y_batch]
            
            yield x_batch_padded, y_batch_padded

# CHANGE 7: RNN class remains the same, but vocabulary is now words instead of characters
class RNN:
    def __init__(self, vocab_size, hidden_size=512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Better initialization for larger vocab
        self.Wxh = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0/vocab_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        self.h_prev = np.zeros((hidden_size, 1))

    def one_hot_encode(self, idx):
        """Convert index to one-hot encoded vector."""
        x = np.zeros((self.vocab_size, 1))
        x[idx] = 1
        return x

    def forward(self, inputs):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.h_prev)
        for t in range(len(inputs)):
            xs[t] = self.one_hot_encode(inputs[t])
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            e_y = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = e_y / np.sum(e_y)
        self.last_cache = (xs, hs, ys, ps)
        self.h_prev = hs[len(inputs) - 1]
        return xs, hs, ys, ps

    def sample(self, seed_idx, length, temperature=1.2, min_length=3):
        x = self.one_hot_encode(seed_idx)
        h = np.copy(self.h_prev)
        output = []
        
        for i in range(length):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            # Mask END_TOKEN if we haven't reached minimum length
            if len(output) < min_length:
                y[word_to_idx[END_TOKEN]] = -np.inf
            
            # Apply temperature
            y = y / temperature
            e_y = np.exp(y - np.max(y))
            p = e_y / np.sum(e_y)
            
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            output.append(idx)
            x = self.one_hot_encode(idx)
            
            if idx == word_to_idx[END_TOKEN]:
                break
                
        return output


    def lossandgrads(self, targets, pad_token=0):
        xs, hs, ys, ps = self.last_cache
        loss = 0.0
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        valid_timesteps = 0

        for t in reversed(range(len(xs))):
            if targets[t] == pad_token:
                continue
            valid_timesteps += 1
            loss += -np.log(ps[t][targets[t], 0] + 1e-9)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, xs[t].T)
            dWhh += np.dot(dh_raw, hs[t - 1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Tighter gradient clipping for word-level
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -1, 1, out=dparam)
        
        self.grads = (dWxh, dWhh, dWhy, dbh, dby)
        return loss / (valid_timesteps if valid_timesteps > 0 else 1)

    def update_params(self, learning_rate=1e-3):
        dWxh, dWhh, dWhy, dbh, dby = self.grads
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                [dWxh, dWhh, dWhy, dbh, dby]):
            param -= learning_rate * dparam

    def train_step(self, inputs, targets, learning_rate=1e-3):
        self.forward(inputs)
        loss = self.lossandgrads(targets)
        self.update_params(learning_rate)
        return loss

# CHANGE 8: Initialize model with word vocabulary size
model = RNN(vocab_size, hidden_size=512)

# CHANGE 9: Updated training loop
batch_gen = batch_generator(X, Y, 32)  # Smaller batch size for word-level

for epoch in range(10000):  # You'll need fewer epochs now!
    x_batch, y_batch = next(batch_gen)
    total_loss = 0
    
    for x, y in zip(x_batch, y_batch):
        model.h_prev = np.zeros((model.hidden_size, 1))  # Reset hidden state
        loss = model.train_step(x, y, learning_rate=1e-3)  # Higher learning rate OK for word-level
        total_loss += loss
    
    if epoch % 20 == 0:
        avg_loss = total_loss / len(x_batch)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # DIAGNOSTIC: Check what the model is predicting
        seed_idx = word_to_idx[START_TOKEN]
        x = model.one_hot_encode(seed_idx)
        h = np.zeros((model.hidden_size, 1))
        
        # Forward pass for first prediction
        h = np.tanh(np.dot(model.Wxh, x) + np.dot(model.Whh, h) + model.bh)
        y = np.dot(model.Why, h) + model.by
        e_y = np.exp(y - np.max(y))
        p = e_y / np.sum(e_y)
        
        # Check top 5 most likely words
        top_indices = np.argsort(p.ravel())[-5:][::-1]
        print("Top 5 predictions after START:")
        for idx in top_indices:
            word = idx_to_word[idx]
            prob = p[idx, 0]
            print(f"  {word}: {prob:.4f}")
        
        # Generate sample
        sample_idxs = model.sample(seed_idx, 10)
        if sample_idxs:
            generated_words = [idx_to_word[i] for i in sample_idxs]
            generated_title = ' '.join(generated_words).replace(START_TOKEN, '').replace(END_TOKEN, '').strip()
            print(f"Sample: '{generated_title}'")
        else:
            print("Sample: EMPTY")
        print("-" * 40)