from nltk.tokenize import RegexpTokenizer
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
import torch
import warnings
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

UNK_CUTOFF=3
UNKNOWN_TOKEN='<unk>'
START_TOKEN='<sos>'
END_TOKEN='eos'
PAD_TOKEN='<pad>'

EMBEDDING_DIM=300
BATCH_SIZE=128
NUM_LABELS=4
HIDDEN_SIZE=300
lrate=0.001
EPOCHS=15

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

df=pd.read_csv('../input/ass3-curr/test.csv')
test_labels=df['Class Index'].tolist()
df=df['Description']
test_sentences=[]
for sent in df:
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(sent)
    tokens=[token.lower() for token in tokens]
    test_sentences.append(tokens)
for sen in test_sentences:
    sen.insert(0,START_TOKEN)
    sen.append(END_TOKEN)

class ELMO(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, out_size):
        super(ELMO, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(hidden_dim, out_size, 1, batch_first=True)
        self.lstm1 = torch.nn.LSTM(hidden_dim, out_size, 1, batch_first=True)
        self.linear = torch.nn.Linear(out_size, vocab_size)
    def forward(self, x):
        embeddings = self.embeddings(x)
        x1, _ = self.lstm(embeddings)
        x2, _ = self.lstm1(x1)
        x = self.linear(x2)
        return x, (embeddings, x1, x2)

forward_model=torch.load('../input/models/forward_model.pt', map_location=torch.device('cuda'))
backward_model=torch.load('../input/models/backward_model.pt', map_location=torch.device('cuda'))
forward_model.eval()
backward_model.eval()
for param in forward_model.parameters():
    param.requires_grad = False
for param in backward_model.parameters():
    param.requires_grad = False

class Dataset_LSTM(Dataset):
  def __init__(self, sent, labs, fm, bm,vocab):
    self.sentences = sent
    self.vocabulary=vocab
    self.labels = labs
    device='cuda'
    self.forward_model=fm.to(device)
    self.backward_model=bm.to(device)
  def __len__(self):
    return len(self.sentences)
  def __getitem__(self, idx):
    sen=[self.vocabulary[w] for w in self.sentences[idx]]
    return torch.tensor(sen),torch.tensor(self.labels[idx]-1)
  def collate(self, batch):
    device='cuda'
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    padded_sentences=pad_sequence(sentences,batch_first=True,padding_value=self.vocabulary[PAD_TOKEN]).to(device)
    sen_rev=torch.flip(padded_sentences,dims=[1]).to(device)
    _, (fe0, fe1, fe2) = self.forward_model(padded_sentences)
    _, (be0, be1, be2) = self.backward_model(sen_rev)
    be0=torch.flip(be0,dims=[1])
    be1=torch.flip(be1,dims=[1])
    be2=torch.flip(be2,dims=[1])
    e0=torch.cat((fe0,be0), dim=2)
    e1=torch.cat((fe1,be1),dim=2)
    e2=torch.cat((fe2,be2),dim=2)
    return torch.tensor(e0), torch.tensor(e1), torch.tensor(e2), torch.tensor(labels)
  
train_dataset=Dataset_LSTM(sentences,train_labels,forward_model,backward_model,vocab)
test_dataset=Dataset_LSTM(test_sentences,test_labels,forward_model,backward_model,vocab)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,collate_fn=train_dataset.collate)
test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=test_dataset.collate)

class LSTMModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, hyp):
        super(LSTMModel, self).__init__()
        if hyp==0:
            self.weight1 = torch.randn(1).to(device)
            self.weight2 = torch.randn(1).to(device)
            self.weight3 = torch.randn(1).to(device)
        else:
            self.weight1 = torch.nn.Parameter(torch.tensor(0.33))
            self.weight2 = torch.nn.Parameter(torch.tensor(0.33))
            self.weight3 = torch.nn.Parameter(torch.tensor(0.33))
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, 1)
        self.hidden2label = torch.nn.Linear(hidden_dim, num_classes, 1)
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2label(lstm_out[-1])
        tag_scores = torch.softmax(tag_space, dim=1)
        return tag_scores

model = LSTMModel(300, 300, NUM_LABELS, 1).to(device)
model=model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lrate)
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for i,(e0,e1,e2,lab) in enumerate(train_dataloader):
        (e0,e1,e2,lab) = (e0.to(device), e1.to(device), e2.to(device), lab.to(device))
        concatenated_params = torch.cat([model.weight1.unsqueeze(0),model.weight2.unsqueeze(0), model.weight3.unsqueeze(0)], dim=0)
        softmax_output = F.softmax(concatenated_params, dim=0)
        softmax_output_list = torch.split(softmax_output, 1)
        temp=(softmax_output_list[0]*e0+softmax_output_list[1]*e1+softmax_output_list[2]*e2)
        outputs = model(temp.permute(1,0,2))
        loss = loss_fn(outputs, lab)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    concatenated_params = torch.cat([model.weight1.unsqueeze(0),model.weight2.unsqueeze(0), model.weight3.unsqueeze(0)], dim=0)
    softmax_output = F.softmax(concatenated_params, dim=0)
    softmax_output_list = torch.split(softmax_output, 1)
    print(softmax_output_list)
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

model.eval()
predictions=[]
true_vals=[]
with torch.no_grad():
    for e0,e1,e2,lab in train_dataloader:
        (e0,e1,e2,lab) = (e0.to(device), e1.to(device), e2.to(device), lab.to(device))
        concatenated_params = torch.cat([model.weight1.unsqueeze(0),model.weight2.unsqueeze(0), model.weight3.unsqueeze(0)], dim=0)
        softmax_output = F.softmax(concatenated_params, dim=0)
        softmax_output_list = torch.split(softmax_output, 1)
        pred = model((softmax_output_list[0]*e0+softmax_output_list[1]*e1+softmax_output_list[2]*e2).permute(1,0,2))
        pred_max_index = torch.argmax(pred, dim=1)
        true_vals.extend(lab.cpu())
        predictions.extend(pred_max_index.cpu())
predictions=torch.stack(predictions).numpy()
true_vals=torch.stack(true_vals).numpy()
print('Evaluation Metrics for train set :')
print(f'Accuracy Score: {accuracy_score(true_vals,predictions)}')
print('F1_Score (Macro)',f1_score(true_vals,predictions, average='macro'))
print('F1_Score (Micro)', f1_score(true_vals,predictions, average='micro'))
print('Precision Score:', precision_score(true_vals,predictions, average='weighted'))
print('Recall Score:',recall_score(true_vals,predictions, average='weighted'))
print('Confusion Matrix:\n',confusion_matrix(true_vals,predictions))

model.eval()
predictions=[]
true_vals=[]
with torch.no_grad():
    for e0,e1,e2,lab in test_dataloader:
        (e0,e1,e2,lab) = (e0.to(device), e1.to(device), e2.to(device), lab.to(device))
        concatenated_params = torch.cat([model.weight1.unsqueeze(0),model.weight2.unsqueeze(0), model.weight3.unsqueeze(0)], dim=0)
        softmax_output = F.softmax(concatenated_params, dim=0)
        softmax_output_list = torch.split(softmax_output, 1)
        pred = model((softmax_output_list[0]*e0+softmax_output_list[1]*e1+softmax_output_list[2]*e2).permute(1,0,2))
        pred_max_index = torch.argmax(pred, dim=1)
        true_vals.extend(lab.cpu())
        predictions.extend(pred_max_index.cpu())
predictions=torch.stack(predictions).numpy()
true_vals=torch.stack(true_vals).numpy()
print('Evaluation Metrics for test set :')
print(f'Accuracy Score: {accuracy_score(true_vals,predictions)}')
print('F1_Score (Macro)',f1_score(true_vals,predictions, average='macro'))
print('F1_Score (Micro)', f1_score(true_vals,predictions, average='micro'))
print('Precision Score:', precision_score(true_vals,predictions, average='weighted'))
print('Recall Score:',recall_score(true_vals,predictions, average='weighted'))
print('Confusion Matrix:\n',confusion_matrix(true_vals,predictions))