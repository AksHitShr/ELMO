from nltk.tokenize import RegexpTokenizer
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

UNK_CUTOFF=3
UNKNOWN_TOKEN='<unk>'
START_TOKEN='<sos>'
END_TOKEN='eos'
PAD_TOKEN='<pad>'

EMBEDDING_DIM=300
BATCH_SIZE=32
HIDDEN_SIZE=300
lrate=0.001
EPOCHS=10

df=pd.read_csv('../input/ass3-curr/train.csv')
train_labels=df['Class Index'].tolist()
df=df['Description']
warnings.filterwarnings("ignore")
sentences=[]
for sent in df:
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(sent)
    tokens=[token.lower() for token in tokens]
    sentences.append(tokens)
for sen in sentences:
    sen.insert(0,START_TOKEN)
    sen.append(END_TOKEN)

def replace_low_frequency_words(sentences, threshold=UNK_CUTOFF):
    word_counts = Counter(word for sentence in sentences for word in sentence)
    replaced_sentences = [
        [UNKNOWN_TOKEN if word_counts[word] < threshold else word for word in sentence]
        for sentence in sentences
    ]
    return replaced_sentences
sentences=replace_low_frequency_words(sentences)
vocab=build_vocab_from_iterator(sentences, specials=[PAD_TOKEN])
vocab.set_default_index(vocab[UNKNOWN_TOKEN])

device = "cuda" if torch.cuda.is_available() else "cpu"
device

class ELMO_Dataset(Dataset):
    def __init__(self, sentences, vocab):
        self.sentences = sentences
        self.vocabulary=vocab
        self.forward_inp, self.forward_out = self.forward()
        self.backward_inp, self.backward_out = self.backward()
    def forward(self):
        inp=[]
        out=[]
        for sent in self.sentences:
            loc_inp = []
            loc_out = []
            for i in range(0, len(sent)-1):
                loc_inp.append(self.vocabulary[sent[i]])
                loc_out.append(self.vocabulary[sent[i+1]])
            inp.append(loc_inp)
            out.append(loc_out)
        return inp,out
    def backward(self):
        inp=[]
        out=[]
        for_inp = self.forward_inp
        for_out = self.forward_out
        for label in for_out:
            rev_label = label[::-1]
            inp.append(rev_label)
        for label in for_inp:
            rev_label = label[::-1]
            out.append(rev_label)
        return inp,out
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return torch.tensor(self.forward_inp[idx]), torch.tensor(self.backward_inp[idx]), torch.tensor(self.forward_out[idx]), torch.tensor(self.backward_out[idx])
    def collate(self,batch):
        for_inp = [i[0] for i in batch]
        bac_inp = [i[1] for i in batch]
        for_out = [i[2] for i in batch]
        bac_out = [i[3] for i in batch]
        for_inp = pad_sequence(for_inp, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        bac_inp = pad_sequence(bac_inp, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        for_out = pad_sequence(for_out, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        bac_out = pad_sequence(bac_out, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        return for_inp,bac_inp,for_out,bac_out

df=pd.read_csv('../input/ass3-curr/test.csv')
test_labels=df['Class Index'].tolist()
df=df['Description']
test_sentences=[]
for sent in df:
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(sent)
    tokens=[token.lower() for token in tokens]
    test_sentences.append(tokens)

train_dataset=ELMO_Dataset(sentences,vocab)
test_dataset=ELMO_Dataset(test_sentences,vocab)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate)
test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=test_dataset.collate)

class ELMO(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, out_size):
        super(ELMO, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm1 = torch.nn.LSTM(hidden_dim, out_size, 1)
        self.lstm2 = torch.nn.LSTM(hidden_dim, out_size, 1)
        self.linear = torch.nn.Linear(out_size, vocab_size)
    def forward(self, x):
        embeddings = self.embeddings(x)
        x1, _ = self.lstm1(embeddings)
        x2, _ = self.lstm2(x1)
        x = self.linear(x2)
        return x, (embeddings, x1, x2)
    
forward_model = ELMO(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, EMBEDDING_DIM).to(device)
backward_model = ELMO(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, EMBEDDING_DIM).to(device)
forward_model = forward_model.to(device)
backward_model = backward_model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(forward_model.parameters(), lr=lrate)
optimizer2 = torch.optim.Adam(backward_model.parameters(), lr=lrate)

loss_for_lst=[]
loss_back_lst=[]

for epoch in range(EPOCHS):
    forward_model.train()
    backward_model.train()
    fwd_loss=0  
    back_loss=0
    for batch, (forward_inp, backward_inp, forward_out, backward_out) in enumerate(train_dataloader):
        (forward_inp, backward_inp, forward_out, backward_out)=(forward_inp.to(device), backward_inp.to(device), forward_out.to(device), backward_out.to(device))
        forward_out = forward_out.view(forward_out.shape[1], forward_out.shape[0])
        backward_out = backward_out.view(backward_out.shape[1], backward_out.shape[0])
        forward_inp = forward_inp.view(forward_inp.shape[1], forward_inp.shape[0])
        backward_inp = backward_inp.view(backward_inp.shape[1], backward_inp.shape[0])
        # forward
        optimizer1.zero_grad()
        forward_out,_  = forward_model(forward_inp)
        forward_out = forward_out.view(forward_out.shape[0]*forward_out.shape[1], forward_out.shape[2])
        forward_out = forward_out.view(forward_out.shape[0]*forward_out.shape[1])
        forward_loss = loss_fn(forward_out, forward_out)
        forward_loss.backward()
        optimizer1.step()
        # backward
        optimizer2.zero_grad()
        backward_out,_ = backward_model(backward_inp)
        backward_out = backward_out.view(backward_out.shape[0]*backward_out.shape[1], backward_out.shape[2])
        backward_out = backward_out.view(backward_out.shape[0]*backward_out.shape[1])
        backward_loss = loss_fn(backward_out, backward_out)
        backward_loss.backward()
        optimizer2.step()
        fwd_loss+=forward_loss
        back_loss+=backward_loss
    loss_for_lst.append(fwd_loss)
    loss_back_lst.append(back_loss)
    print(f'EPOCH: {epoch}, Forward Loss: {fwd_loss}, Backward Loss: {back_loss}')


torch.save(forward_model,'forward_model.pt')
torch.save(backward_model,'backward_model.pt')

a=[float(i.cpu().detach().numpy()) for i in loss_for_lst]
b=[float(i.cpu().detach().numpy()) for i in loss_back_lst]

plt.plot(list(range(1,11)), a, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss on train dataset')
plt.title('Epoch vs Loss for forward Model')
plt.show()

plt.plot(list(range(1,11)), b, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss on train dataset')
plt.title('Epoch vs Loss for backward Model')
plt.show()