from __future__ import print_function
import sys, os, re, random, csv
import os.path
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from IPython.display import display, HTML

class DataReader:
    def __init__(self, data_file, num_meta_cols, multi_pass):
        self.num_meta_cols = num_meta_cols
        self.multi_pass = multi_pass
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')
        self.__load_vocab()
        self.__load_idfs()
        self.__init_data(data_file)
        self.__allocate_minibatch()

    def __tokenize(self, s, max_terms):
        return self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', s.lower())).strip().split()[:max_terms]

    def __load_vocab(self):
        global VOCAB_SIZE
        self.vocab = {}
        with open(DATA_FILE_VOCAB, mode='r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.vocab[row[0]] = int(row[1])
        VOCAB_SIZE = len(self.vocab) + 1
        embeddings = np.zeros((VOCAB_SIZE, NUM_HIDDEN_NODES), dtype=np.float32)
        with open(DATA_EMBEDDINGS, mode='r', encoding="utf-8") as f:
            for line in f:
                cols = line.split()
                idx = self.vocab.get(cols[0], 0)
                if idx > 0:
                    for i in range(NUM_HIDDEN_NODES):
                        embeddings[idx, i] = float(cols[i + 1])
        self.pre_trained_embeddings = torch.tensor(embeddings)
        
    def __load_idfs(self):
        self.idfs = {}
        with open(DATA_FILE_IDFS, mode='r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.idfs[row[0]] = float(row[1])

    def __init_data(self, file_name):
        self.reader = open(file_name, mode='r', encoding="utf-8")
        self.num_docs = len(self.reader.readline().split('\t')) - self.num_meta_cols - 1
        self.reader.seek(0)
    
    def __allocate_minibatch(self):
        self.features = {}
        if ARCH_TYPE != 1:
            self.features['local'] = [] 
        if ARCH_TYPE > 0:
            self.features['dist_q'] = np.zeros((MB_SIZE, MAX_QUERY_TERMS), dtype=np.int64)
            self.features['mask_q'] = np.zeros((MB_SIZE, MAX_QUERY_TERMS, 1), dtype=np.float32)
            self.features['dist_d'] = []
            self.features['mask_d'] = []
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS, MAX_QUERY_TERMS), dtype=np.float32))
            if ARCH_TYPE > 0:
                self.features['dist_d'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS), dtype=np.int64))
                self.features['mask_d'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS, 1), dtype=np.float32))
        self.features['labels'] = np.zeros((MB_SIZE), dtype=np.int64)
        self.features['meta'] = []
        
    def __clear_minibatch(self):
        if ARCH_TYPE > 0:
            self.features['dist_q'].fill(np.int64(0))
            self.features['mask_q'].fill(np.float32(0))
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'][i].fill(np.float32(0))
            if ARCH_TYPE > 0:
                self.features['dist_d'][i].fill(np.int64(0))
                self.features['mask_d'][i].fill(np.float32(0))
        self.features['meta'].clear()

    def get_minibatch(self):
        self.__clear_minibatch()
        for i in range(MB_SIZE):
            row = self.reader.readline()
            if row == '':
                if self.multi_pass:
                    self.reader.seek(0)
                    row = self.reader.readline()
                else:
                    break
            cols = row.split('\t')
            q = self.__tokenize(cols[self.num_meta_cols], MAX_QUERY_TERMS)
            ds = [self.__tokenize(cols[self.num_meta_cols + i + 1], MAX_DOC_TERMS) for i in range(self.num_docs)]
            if ARCH_TYPE != 1:
                for d in range(self.num_docs):
                    for j in range(len(ds[d])):
                        for k in range(len(q)):
                            if ds[d][j] == q[k]:
                                self.features['local'][d][i, j, k] = self.idfs[q[k]]
            if ARCH_TYPE > 0:
                for j in range(self.num_docs + 1):
                    terms = q if j == 0 else ds[j - 1]
                    for t in range(len(terms)):
                        term = self.vocab.get(terms[t], 0)
                        if j == 0:
                            self.features['dist_q'][i, t] = term
                            self.features['mask_q'][i, t, 0] = 1
                        else:
                            self.features['dist_d'][j - 1][i, t] = term
                            self.features['mask_d'][j - 1][i, t, 0] = 1
            self.features['meta'].append(tuple(cols[:self.num_meta_cols]))
        return self.features
    
    def reset(self):
        self.reader.seek(0)


display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))

def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

## Model

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class Duet(torch.nn.Module):
    
    def __init__(self):
        super(Duet, self).__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, NUM_HIDDEN_NODES)
        self.embed.weight = nn.Parameter(READER_TRAIN.pre_trained_embeddings, requires_grad=True)
        self.duet_local = nn.Sequential(nn.Conv1d(MAX_DOC_TERMS, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * MAX_QUERY_TERMS, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_dist_q = nn.Sequential(nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(POOLING_KERNEL_WIDTH_QUERY),
                                        Flatten(),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU()
                                        )
        self.duet_dist_d = nn.Sequential(nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(POOLING_KERNEL_WIDTH_DOC, stride=1),
                                        nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.ReLU()
                                        )
        self.duet_dist = nn.Sequential(Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * NUM_POOLING_WINDOWS_DOC, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_comb = nn.Sequential(nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, 1),
                                        nn.ReLU())
        self.scale = torch.tensor([0.1], requires_grad=False).to(DEVICE)

    def forward(self, x_local, x_dist_q, x_dist_d, x_mask_q, x_mask_d):
        if ARCH_TYPE != 1:
            h_local = self.duet_local(x_local)
        if ARCH_TYPE > 0:
            h_dist_q = self.duet_dist_q((self.embed(x_dist_q) * x_mask_q).permute(0, 2, 1))
            h_dist_d = self.duet_dist_d((self.embed(x_dist_d) * x_mask_d).permute(0, 2, 1))
            h_dist = self.duet_dist(h_dist_q.unsqueeze(-1)*h_dist_d)
        y_score = self.duet_comb((h_local + h_dist) if ARCH_TYPE == 2 else (h_dist if ARCH_TYPE == 1 else h_local)) * self.scale
        return y_score
    
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

## Settings
DEVICE = torch.device("cuda:0")    # torch.device("cpu")
ARCH_TYPE = 2
MAX_QUERY_TERMS = 20
MAX_DOC_TERMS = 200
NUM_HIDDEN_NODES = 300
TERM_WINDOW_SIZE = 3
POOLING_KERNEL_WIDTH_QUERY = MAX_QUERY_TERMS - TERM_WINDOW_SIZE + 1 # 20 - 3 + 1 = 18
POOLING_KERNEL_WIDTH_DOC = 100
NUM_POOLING_WINDOWS_DOC = (MAX_DOC_TERMS - TERM_WINDOW_SIZE + 1) - POOLING_KERNEL_WIDTH_DOC + 1 # (200 - 3 + 1) - 100 + 1 = 99
VOCAB_SIZE = 0
DROPOUT_RATE = 0.5
MB_SIZE = 1024
EPOCH_SIZE = 1024
NUM_EPOCHS = 1
NUM_ENSEMBLES = 8
LEARNING_RATE = 0.001
DATA_DIR = 'data/'
DATA_FILE_VOCAB = os.path.join(DATA_DIR, "word-vocab-small.tsv")
DATA_EMBEDDINGS = os.path.join(DATA_DIR, "glove.6B\\glove.6B.{}d.txt".format(NUM_HIDDEN_NODES))
DATA_FILE_IDFS = os.path.join(DATA_DIR, "idfnew.norm.tsv")
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "triples.train.full.tsv")
DATA_FILE_DEV = os.path.join(DATA_DIR, "top1000.dev.tsv")
DATA_FILE_EVAL = os.path.join(DATA_DIR, "top1000.eval.tsv")
QRELS_DEV = os.path.join(DATA_DIR, "qrels.dev.tsv")
DATA_FILE_OUT_DEV = os.path.join(DATA_DIR, "output.dev.tsv")
DATA_FILE_OUT_EVAL = os.path.join(DATA_DIR, "output.eval.tsv")
MODEL_FILE = os.path.join(DATA_DIR, "duet.ens{}.ep{}.dnn")
READER_TRAIN = DataReader(DATA_FILE_TRAIN, 0, True)
READER_DEV = DataReader(DATA_FILE_DEV, 2, False)
READER_EVAL = DataReader(DATA_FILE_EVAL, 2, False)

qrels = {}
with open(QRELS_DEV, mode='r', encoding="utf-8") as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        qid = int(row[0])
        did = int(row[2])
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(did)

res_dev = {}
res_eval = {}

print_message('Starting')
print_message('Learning rate: {}'.format(LEARNING_RATE))

for ens_idx in range(NUM_ENSEMBLES):
    torch.manual_seed(ens_idx + 1)
    net = Duet()
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    print_message('Number of learnable parameters: {}'.format(net.parameter_count()))
    for ep_idx in range(NUM_EPOCHS):
        train_loss = 0.0
        net.train()
        for mb_idx in range(EPOCH_SIZE):
            features = READER_TRAIN.get_minibatch()
            if ARCH_TYPE == 0:
                out = torch.cat(tuple([net(torch.from_numpy(features['local'][i]).to(DEVICE), None, None) for i in range(READER_TRAIN.num_docs)]), 1)
            elif ARCH_TYPE == 1:    
                out = torch.cat(tuple([net(None, torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][i]).to(DEVICE), torch.from_numpy(features['mask_q']).to(DEVICE), torch.from_numpy(features['mask_d'][i]).to(DEVICE)) for i in range(READER_TRAIN.num_docs)]), 1)
            else:
                out = torch.cat(tuple([net(torch.from_numpy(features['local'][i]).to(DEVICE), torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][i]).to(DEVICE), torch.from_numpy(features['mask_q']).to(DEVICE), torch.from_numpy(features['mask_d'][i]).to(DEVICE)) for i in range(READER_TRAIN.num_docs)]), 1)
            loss = criterion(out, torch.from_numpy(features['labels']).to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        torch.save(net, MODEL_FILE.format(ens_idx + 1, ep_idx + 1))
        print_message('model:{}, epoch:{}, loss:{}'.format(ens_idx + 1, ep_idx + 1, train_loss / EPOCH_SIZE))
    is_complete = False
    READER_DEV.reset()
    net.eval()

    while not is_complete:
        features = READER_DEV.get_minibatch()
        if ARCH_TYPE == 0:
            out = net(torch.from_numpy(features['local'][0]).to(DEVICE), None, None)
        elif ARCH_TYPE == 1:        
            out = net(None, torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][0], torch.from_numpy(features['mask_q']).to(DEVICE), torch.from_numpy(features['mask_d'][0]).to(DEVICE)).to(DEVICE))
        else:
            out = net(torch.from_numpy(features['local'][0]).to(DEVICE), torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][0]).to(DEVICE), torch.from_numpy(features['mask_q']).to(DEVICE), torch.from_numpy(features['mask_d'][0]).to(DEVICE))
        meta_cnt = len(features['meta'])
        out = out.data.cpu()
        for i in range(meta_cnt):
            q = int(features['meta'][i][0])
            d = int(features['meta'][i][1])
            if q not in res_dev:    
                res_dev[q] = {}
            if d not in res_dev[q]:
                res_dev[q][d] = 0
            res_dev[q][d] += out[i][0]
        is_complete = (meta_cnt < MB_SIZE)
    is_complete = False
    READER_EVAL.reset()
    net.eval()
    while not is_complete:
        features = READER_EVAL.get_minibatch()
        if ARCH_TYPE == 0:
            out = net(torch.from_numpy(features['local'][0]).to(DEVICE), None, None)
        elif ARCH_TYPE == 1:
            out = net(None, torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][0], torch.from_numpy(features['mask_q']).to(DEVICE), torch.from_numpy(features['mask_d'][0]).to(DEVICE)).to(DEVICE))
        else:
            out = net(torch.from_numpy(features['local'][0]).to(DEVICE), torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][0]).to(DEVICE), torch.from_numpy(features['mask_q']).to(DEVICE), torch.from_numpy(features['mask_d'][0]).to(DEVICE))
        meta_cnt = len(features['meta'])
        out = out.data.cpu()
        for i in range(meta_cnt):
            q = int(features['meta'][i][0])
            d = int(features['meta'][i][1])
            if q not in res_eval:
                res_eval[q] = {}
            if d not in res_eval[q]:
                res_eval[q][d] = 0
            res_eval[q][d] += out[i][0]
        is_complete = (meta_cnt < MB_SIZE)
    mrr = 0
    for qid, docs in res_dev.items():
        ranked = sorted(docs, key=docs.get, reverse=True)
        for i in range(min(len(ranked), 10)):
            if ranked[i] in qrels[qid]:
                mrr += 1 / (i + 1)
                break
    mrr /= len(qrels)
    print_message('model:{}, mrr:{}'.format(ens_idx + 1, mrr))

with open(DATA_FILE_OUT_DEV, mode='w', encoding="utf-8") as f:
    for qid, docs in res_dev.items():
        ranked = sorted(docs, key=docs.get, reverse=True)
        for i in range(min(len(ranked), 10)):
            f.write('{}\t{}\t{}\n'.format(qid, ranked[i], i + 1))

with open(DATA_FILE_OUT_EVAL, mode='w', encoding="utf-8") as f:
    for qid, docs in res_eval.items():
        ranked = sorted(docs, key=docs.get, reverse=True)
        for i in range(min(len(ranked), 10)):
            f.write('{}\t{}\t{}\n'.format(qid, ranked[i], i + 1))
print_message('Completed')

