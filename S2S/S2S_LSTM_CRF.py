import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import Seq2SeqUtility as s2s
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
from DecisionTreeFinder import ParaphraseData, write_train_validation_test,feature_file_to_x_y_sentences,\
    from_file_to_pd_list
import pickle
import tqdm
# from CRF_TEST import CRF
from torchcrf import CRF
# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300


ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

#------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump([obj], f,-1)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


#------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :return:
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_for_curpos = {k: full_w2v[k] for k in words_list if k in full_w2v}
        w2v_emb_dict = EmbeddingDict(w2v_for_curpos, W2V_EMBEDDING_DIM)
        save_pickle(w2v_emb_dict,w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict

def get_sentence_average_w2v(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector.
    """
    sum_vec = np.zeros((embedding_dim,))
    known_tokens = 0
    for token in sent.text:
        if (token in word_to_vec.dict):
            known_tokens += 1
            sum_vec += word_to_vec[token]
    if (known_tokens != 0):
        return sum_vec / known_tokens
    else:
        return sum_vec


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return:
    """
    one_hot = np.zeros((size,))
    one_hot[ind] = 1
    return one_hot

def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    known_words = 0
    size = len(word_to_ind.keys())
    sum_vec = np.zeros((size,))
    for token in sent.text: #going over all tokens and summing their embeddings
        if (token in word_to_ind):
            sum_vec += get_one_hot(size, word_to_ind[token])
            known_words += 1
    if (known_words != 0):
        return sum_vec / known_words
    else:
        return sum_vec

def get_word_to_ind (count_dict, cutoff = 0):
    """
    this function gets a dictionary mapping words to their counts, and returns a mapping between
    words to their index. Words that come after the cutoff
    :param count_dict: a dictionary, mapping words to their index
    :return:
    """
    word_to_ind = {}
    if (not cutoff):
        cutoff = len(count_dict.keys())
    #sorting the words by their count:
    sorted_tuples = list(reversed(sorted(count_dict.items(), key=operator.itemgetter(1))))
    for i in range(cutoff):
        cur_word = sorted_tuples[i][0]
        word_to_ind[cur_word] = i

    return word_to_ind


def sent_to_indices(sent,word2ind,seq_len):
    """
    :param sent:
    :param seq_len:
    :return:
    """
    sent_ind = []
    sent_len = len(sent.text)
    for i in range(min(sent_len,seq_len)):
        token = sent.text[i]
        sent_ind.append(word2ind.get(token))
    #padding
    if (sent_len < seq_len):
        for j in range(seq_len - sent_len):
            sent_ind.append(word2ind.range)
    return sent_ind

def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :return:
    """
    embedding_vec = np.zeros((seq_len,embedding_dim))
    for i in range(min(len(sent),seq_len)):
        embedding_vec[i,:] = word_to_vec.get(sent[i])
    return embedding_vec


class EmbeddingDict:
    def __init__(self, dict, embedding_size):
        self.dict = dict
        self.unkown_vec = np.zeros((embedding_size,))
    def get (self,key):
        if (key in self.dict):
            return self.dict[key]
        else:
            return self.unkown_vec
    def __getitem__(self, key):
        return self.get(key)


#------------------------------------ Models ----------------------------------------------------



class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        """
        building the encoder
        :param embedding_dim:
        :param hidden_dim:
        :param n_layers:
        :param dropout:
        """
        super(Encoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, input_features):
        """

        :param input_features:
        :return:
        """
        input_features = input_features.permute(1,0,2)
        _, (hidden,_) = self.LSTM(input_features)
        return torch.cat((hidden[-2,:,:].T,hidden[-1,:,:].T)).T


class DecoderCRF(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderCRF, self).__init__()
        self.LSTM = nn.LSTM(input_size=1, hidden_size=2*hidden_dim)
        self.hidden2tag = nn.Linear(2*hidden_dim,2)
        self.crf_layer = CRF(2)

    def forward(self, input_features, hidden,tags,masks):
        """

        :param input_features: float in [0,1]
        :param hidden: the last hidden state
        :return:
        """
        tags = tags.permute(1, 0, 2)
        tags = tags.squeeze()
        tags = torch.tensor(tags, dtype=torch.long)
        masks = masks.squeeze()
        masks = torch.tensor(masks, dtype=torch.uint8)
        seq_emb = input_features.permute(1, 0, 2)
        hidden, _ = self.LSTM(input_features,(hidden,torch.zeros(hidden.size())))
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        likelihood = self.crf_layer(emission, tags, masks)
        # hidden1_full = self.dropout(packed_output1)
        return -likelihood

    def predict(self, seq_emb):
        seq_emb = seq_emb.permute(1, 0, 2)
        hidden, _ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        return torch.tensor(self.crf_layer.decode(emission))

class LSTMcrf(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):

        super().__init__()
        self.crf_layer = CRF(2)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,num_layers=n_layers)
        self.hidden2ta = nn.Linear(2*hidden_dim,2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, seq_emb,tags,masks):
        tags = tags.permute(1,0,2)
        tags = tags.squeeze()
        tags = torch.tensor(tags,dtype=torch.long)
        masks = masks.squeeze()
        masks = torch.tensor(masks,dtype=torch.uint8)
        seq_emb = seq_emb.permute(1,0,2)
        hidden,_ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        likelihood  = self.crf_layer(emission,tags,masks)
        # hidden1_full = self.dropout(packed_output1)
        return -likelihood

    def predict(self, seq_emb):
        seq_emb = seq_emb.permute(1, 0, 2)
        hidden, _ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        return torch.tensor(self.crf_layer.decode(emission))




#------------------------- training functions -------------


def binary_accuracy(preds, y, pr=False):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    ons = torch.where(y == 1)
    ons_correct = (rounded_preds[ons]==y[ons]).float()
    ons_acc = ons_correct.sum() / ons[0].shape[0]
    correct = (rounded_preds == y).float() #convert into float for division
    all_corrects = [1 if torch.all(rounded_preds[:,i] ==y[:,i]) else 0 for i in range(y.size(1))]
    all_correct_acc = sum(all_corrects) / len(all_corrects)
    acc = correct.sum() / (correct.shape[0]*correct.shape[1])
    # if (pr):
    #     print(rounded_preds)
    #     print(y)
    return acc, ons_acc, all_correct_acc

def train_encoder_decoder_epoch(encoder,decoder, data_iterator, enc_optimizer,dec_optimizer,criterion,size):

    epoch_loss = 0
    epoch_acc = 0
    epoch_ons_acc = 0
    epoch_all_correct_acc = 0
    encoder.train()
    decoder.train()
    num_examples = 0
    for batch in tqdm.tqdm(data_iterator):
        num_examples += 64
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        x = batch[0]
        y = batch[1]
        y = y.permute(1, 0, 2)
        target_length = y.size(0)
        # y_pred = torch.zeros()
        context = encoder(x)
        #getting the predictions from  decoder:
        decoder_hidden = context.unsqueeze(0)
        preds = torch.zeros(tuple(y.size()))
        batch_size = y.size(1)
        last_preds = torch.zeros((1,y.size(1),1))
        for i in range(target_length):
            cur_out, decoder_hidden = decoder(last_preds, decoder_hidden)
            last_preds = cur_out
            preds[i]  = cur_out
        loss = criterion(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

        # loss = criterion(predictions.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

        acc, ons_acc, all_correct_acc = binary_accuracy(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()

        epoch_loss += loss.item()*batch_size
        epoch_acc += acc.item()*batch_size
        epoch_ons_acc += ons_acc.item()*batch_size
        epoch_all_correct_acc += all_correct_acc*batch_size

    return epoch_loss / num_examples, epoch_acc / num_examples, epoch_ons_acc / num_examples, epoch_all_correct_acc / num_examples

def train_epoch(model, data_iterator, optimizer,criterion,size):

    epoch_loss = 0
    epoch_acc = 0
    epoch_ons_acc = 0
    epoch_all_correct_acc = 0
    model.train()
    num_examples = 0
    for batch in tqdm.tqdm(data_iterator):
        optimizer.zero_grad()
        x = batch[0]
        y = batch[1]
        batch_size = y.size(0)
        num_examples += batch_size
        mask = batch[2]
        mask = mask.permute(1,0)
        # y_pred = torch.zeros()
        loss = model(x,y,mask)
        preds = model.predict(x)
        y = y.squeeze()
        # loss = criterion(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

        # loss = criterion(predictions.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

        acc, ons_acc, all_correct_acc = binary_accuracy(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()*batch_size
        epoch_acc += acc.item()*batch_size
        epoch_ons_acc += ons_acc.item()*batch_size
        epoch_all_correct_acc += all_correct_acc*batch_size

    return epoch_loss / num_examples, epoch_acc / num_examples, epoch_ons_acc / num_examples, epoch_all_correct_acc / num_examples


def evaluate_encoder_decoder(encoder,decoder, data_iterator,criterion):

    epoch_loss = 0
    epoch_acc = 0
    epoch_ons_acc = 0
    epoch_all_correct_acc = 0
    encoder.train()
    decoder.train()
    num_examples = 0
    with torch.no_grad():
        pr = True
        for batch in tqdm.tqdm(data_iterator):
            num_examples += 64
            x = batch[0]
            y = batch[1]
            y = y.permute(1, 0, 2)
            target_length = y.size(0)
            # y_pred = torch.zeros()
            context = encoder(x)
            #getting the predictions from  decoder:
            decoder_hidden = context.unsqueeze(0)
            preds = torch.zeros(tuple(y.size()))
            batch_size = y.size(1)
            last_preds = torch.zeros((1,y.size(1),1))
            # if pr:
            #     print(torch.round(F.sigmoid(last_preds)))
            #     pr = False
            for i in range(target_length):
                cur_out, decoder_hidden = decoder(last_preds, decoder_hidden)
                last_preds = cur_out
                preds[i]  = cur_out
            loss = criterion(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

            # loss = criterion(predictions.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

            acc, ons_acc, all_correct_acc = binary_accuracy(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor))

            # loss.backward()
            epoch_loss += loss.item()*batch_size
            epoch_acc += acc.item()*batch_size
            epoch_ons_acc += ons_acc.item()*batch_size
            epoch_all_correct_acc += epoch_all_correct_acc*batch_size

    return epoch_loss / num_examples, epoch_acc / num_examples, epoch_ons_acc / num_examples, epoch_all_correct_acc / num_examples

def evaluate(model, data_iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_ons_acc = 0
    epoch_all_acc = 0
    model.eval()
    num_examples = 0

    with torch.no_grad():
        pr = True
        for batch in data_iterator:
            x = batch[0]
            y = batch[1]
            batch_size = y.size(0)
            num_examples += batch_size
            mask = batch[2]
            mask = mask.permute(1, 0)
            # y_pred = torch.zeros()
            loss = model(x, y, mask)
            preds = model.predict(x)
            y = y.squeeze()

            acc, ons_acc,all_acc = binary_accuracy(preds.type(torch.DoubleTensor), y.type(torch.DoubleTensor), pr)
            pr = False

            epoch_loss += loss.item() * 64
            epoch_acc += acc.item() * 64
            epoch_ons_acc += ons_acc.item()*64
            epoch_all_acc += all_acc
    return epoch_loss / num_examples, epoch_acc / num_examples, epoch_ons_acc / num_examples, epoch_all_acc / num_examples


def save_model(model,path, epoch, optimizer):
    """
    :param model:
    :param path:
    :return:
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)

def load(model, path, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model,optimizer,epoch


def get_predictions_for_data(model, data_iter):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in data_iter:
            x,y = batch[0].type(torch.FloatTensor), batch[1]
            pred = (model.predict(x).cpu().numpy() > 0.5).astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds)


def train_encoder_decoder_full(encoder,decoder, train_dl,val_dl,test_dl,n_epochs, lr,train_size,val_size, weight_decay=0., save_model_path=None, \
                                                                                      save_model_freq=1,
                                  save_stats_path=None):
    encoder.to(get_available_device())
    decoder.to(get_available_device())
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(get_available_device())
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_ons_acc":[],"val_ons_acc":[],
               "all_correct_train":[], "all_correct_val":[]}

    for ep in range(n_epochs):
        train_loss, train_acc, train_ons_acc, all_correct_train = train_encoder_decoder_epoch(encoder,decoder, train_dl, encoder_optimizer,
                                            decoder_optimizer,criterion,train_size)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["train_ons_acc"].append(train_ons_acc)
        metrics["all_correct_train"].append(all_correct_train)
        val_loss, val_acc, val_ons_Acc, all_correct_val = evaluate_encoder_decoder(encoder,decoder, val_dl, criterion)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_ons_acc"].append(val_ons_Acc)
        metrics["all_correct_val"].append(all_correct_val)
        # if ((ep + 1) % save_model_freq == 0) and (save_model_path is not None):
        #     save_model(model, save_model_path.format(epoch=ep), ep, optimizer)
        print("\t".join(["{}: {}".format(k, v[-1]) for k,v in metrics.items()]))
    if save_stats_path is not None:
        save_pickle(metrics, save_stats_path)
    test_measures = evaluate_encoder_decoder(encoder,decoder,test_dl,criterion)
    print("------------------------")
    print("------------------------")
    print("------------------------")
    return [metrics["val_loss"],metrics["val_acc"], metrics["val_ons_acc"],metrics["all_correct_val"]],test_measures



def train_model(model, train_dl,val_dl,test_dl,n_epochs, lr,train_size,val_size, weight_decay=0., save_model_path=None, \
                                                                                      save_model_freq=1,
                                  save_stats_path=None):
    model.to(get_available_device())
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(get_available_device())
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_ons_acc":[],"val_ons_acc":[],
               "train_all_acc":[],"val_all_acc":[]}

    for ep in range(n_epochs):
        train_loss, train_acc, train_ons_acc, train_all_correct_acc = train_epoch(model, train_dl, optimizer,
                                            criterion,train_size)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["train_ons_acc"].append(train_ons_acc)
        metrics["train_all_acc"].append(train_all_correct_acc)
        val_loss, val_acc, val_ons_Acc, val_all_acc = evaluate(model, val_dl, criterion)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_ons_acc"].append(val_ons_Acc)
        metrics["val_all_acc"].append(val_all_acc)
        if ((ep + 1) % save_model_freq == 0) and (save_model_path is not None):
            save_model(model, save_model_path.format(epoch=ep), ep, optimizer)
        print("\t".join(["{}: {}".format(k, v[-1]) for k,v in metrics.items()]))
    if save_stats_path is not None:
        save_pickle(metrics, save_stats_path)
    test_measure = evaluate(model,test_dl,criterion)
    return [metrics["val_loss"][-1], metrics["val_acc"][-1],metrics["val_ons_acc"][-1], metrics["val_all_acc"][-1]], test_measure




def get_data_ready(fname,train_perc,val_perc, train_name,val_name,test_name):
    pd = ParaphraseData(fname, normalized = True)
    write_train_validation_test(pd,train_perc,val_perc,train_name,val_name,test_name)

def train_lstm_crf_with_w2v(train_data, val_data, test_data, n_epochs,lr, weight_decay, n_layers, hidden_dim):
    batch_size = 64
    dropout = 0.5
    train_x,train_y = train_data
    val_x,val_y = val_data
    test_x, test_y = test_data
    train_dataloader,train_size = create_dataloader(train_x, train_y,batch_size)
    test_dataloader,test_size = create_dataloader(test_x, test_y,batch_size)
    val_dataloader,val_size = create_dataloader(val_x, val_y,batch_size)
    model = LSTMcrf(embedding_dim=EMB_SIZE, hidden_dim=hidden_dim, n_layers=n_layers,dropout=0.5)
    return train_model(model, train_dataloader,val_dataloader,test_dataloader,train_size=train_size, val_size=val_size, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay), model

def train_encoder_decoder_crf_general(train_data, val_data, test_data, n_epochs,lr, weight_decay, n_layers, hidden_dim):
    batch_size = 64
    dropout = 0.5
    train_x,train_y = train_data
    val_x,val_y = val_data
    test_x, test_y = test_data
    train_dataloader,train_size = create_dataloader(train_x, train_y,batch_size)
    test_dataloader,test_size = create_dataloader(test_x, test_y,batch_size)
    val_dataloader,val_size = create_dataloader(val_x, val_y,batch_size)
    encoder = Encoder(embedding_dim=EMB_SIZE, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    decoder = Decoder(hidden_dim)
    return train_encoder_decoder_full(encoder,decoder, train_dataloader,val_dataloader,test_dataloader,train_size=train_size, val_size=val_size, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay)


class seq2seqDATA(torch.utils.data.IterableDataset):
    def __init__(self,x,y,seq_len):
        self.x = x
        self.y = y
        self.seq_len = seq_len

    def __iter__(self):
        for i in range(len(self.x)):
            try:
                y,mask = self.pad_sequence(self.y[i], 1,True)
            except:
                nir = 1
            yield (torch.FloatTensor(self.pad_sequence(np.array(self.x[i]), EMB_SIZE)),y,mask)

    def __len__(self):
        return len(self.x)

    def pad_sequence(self,seq,emb_size,return_mask=False):
        """

        :param seq:
        :param seq_len:
        :return:
        """
        seq = np.array(seq)
        to_return = np.zeros((self.seq_len, emb_size))
        for i in range(min(self.seq_len, seq.shape[0])):
            to_return[i] = seq[i]
        if (return_mask):
            mask = np.ones(self.seq_len)
            if (self.seq_len > seq.shape[0]):
                for i in range(seq.shape[0],self.seq_len):
                    mask[i] = 0
            return to_return,mask

        return to_return


EMB_SIZE= 704

def create_dataloader(x,y,batch_size):
    # x,y, orig_sents, test_sents = feature_file_to_x_y_sentences(fname)
    #creating the w2v embedding of the sentences:
    seq_len = 15
    # x = pad_sequence(x,seq_len)
    return DataLoader(seq2seqDATA(x,y,seq_len),batch_size=batch_size), len(x)

def get_words_list(pd_list, words_list):
    for dp in pd_list:
        orig_sent1,orig_sent2 = dp.get_original_sentences()
        add_words_to_set(words_list,orig_sent1)
        add_words_to_set(words_list,orig_sent2)
    return words_list

def add_words_to_set(word_set,sent):
    for word in sent.split():
        word_set.add(word)



def crf_exmaple_to_snowclone(model,example,w2v):
    fg = s2s.FeatureGetter(example,w2v)
    it = DataLoader(seq2seqDATA([list(fg)], [[0]*len(example.get_sent_words())], 15), batch_size=1)
    ind_to_new_sent = {}
    for batch in it:
        x = batch[0]
        preds = model.predict(x)
        # for k in all_preds[0]:
        #     preds = all_preds[0][
        cur_tags = [int(x.item()) for x in list(preds[0])]
        new_sent_words = []
        orig_sent_words = example.get_sent_words()
        i = 0
        j = 0
        while (i< min(len(orig_sent_words), 15)):
            try:
                if cur_tags[i] == 1:
                    if (j==0 or new_sent_words[j-1] != "*"):
                        new_sent_words.append("*")
                        j+=1
                    else:
                        i += 1
                else:
                    new_sent_words.append(orig_sent_words[i])
                    i += 1
                    j+=1
            except:
                print("wtfffffffffffffff")
                stop = 1
            # ind_to_new_sent[k] = new_sent_words[:]
    return new_sent_words

    # features = torch.tensor(list(fg), dtype=torch.double)
    # features = features.unsqueeze(0)
    # features.type(torch.DoubleTensor)
    # features = features.double()
    # preds = model.predict(features)

# def write_to_new(sent_path,new_path,model):
#     all_sentences = []
#     with open(sent_path) as f1:
#         l = f1.readline()
#         while (l):
#             cur_l =
#             all_sentences.append()



import  TestS2SModel as test
# if __name__ == '__main__':
#     import Seq2SeqUtility as s2s
#
#     # seq = torch.from_numpy(np.random.normal(0,10,size=(3,10,4)))
#     # seq = seq.type(torch.FloatTensor)
#     # lstm  = LSTM(4,6,1,0.5,0)
#     # lstm.forward(seq)
#     # fname = "paraphrase_data_0306"
#     # train_feature_fname = "paraphrase_features\\train_para_features_idf"
#     # val_feature_fname = "paraphrase_features\\val_para_features_idf"
#     # test_feature_fname = "paraphrase_features\\test_para_features_idf"
#     # pd = ParaphraseData(fname, normalized=True)
#     # # write_train_validation_test(pd,0.6,0.2,train_feature_fname,val_feature_fname,test_feature_fname)
#     # train_pd_list = from_file_to_pd_list(train_feature_fname)
#     # test_pd_list = from_file_to_pd_list(test_feature_fname)
#     # val_pd_list = from_file_to_pd_list(val_feature_fname)
#     # words_list = set()
#     # words_list = get_words_list(train_pd_list,words_list)
#     # words_list = get_words_list(val_pd_list,words_list)
#     # words_list = get_words_list(test_pd_list,words_list)
#     # word_to_vec = create_or_load_slim_w2v(words_list)
#     # train_lstm_with_w2v(train_feature_fname,val_feature_fname,test_feature_fname,word_to_vec)
#     # # nir = create_dataloader(train_feature_fname,word_to_vec)
#     # train_log_linear_with_one_hot()
#     # train_log_linear_with_w2v()
#
#     snowclone_db_path = "patterns_db_test"
#     w2v_path = ""
#     sp_reader = s2s.SentencePatternReader(snowclone_db_path)
#     w2v = s2s.get_w2v("snowclone_w2v.pkl", sp_reader, should_create=False)
#     # print(w2v["START_WORD_STR"])
#     # getting train-val-test split
#     train_perc = 0.7
#     val_perc = 0.15
#     n_epocs = 10
#     lr = 0.01
#     weight_decay = 0
#     n_layers = 2
#     hidden_dim = 32
#     k = 1
#     ch_path  = "C:\\Users\\User\\Desktop\\nir\\Desktop\\Masters\\Second\\Lab\\chrome_server\\"
#     # sp_reader.train_val_test_split(train_perc, val_perc)
#     # train, val, test = sp_reader.get_train_val_test_X_y(w2v)
#     # _, model = train_lstm_crf_with_w2v(train,val,test,n_epocs,lr,weight_decay,n_layers,hidden_dim,k=k)
#     # save_pickle(model,"lstmcrf_k1_model")
#     model = load_pickle("lstmcrf_k1new_model")[0]
#     # save_pickle(model, "lstmcrf_k1new_model")
#     model_tester = test.ModelChecker(model, "CRF", ch_path+"sentences")
#     model_tester.write_model_examples_to_file(ch_path+"patterns",model)
#     # sent = "all nir and no play makes nir a dull boy"
#     # example = s2s.PhraseExample(sent,[])
#     # crf_exmaple_to_snowclone(model,example,w2v)
#
#
# #important formula:
# #torch.round(F.sigmoid(self.hidden2ta(packed_output1)))[:,1,:]