import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForTokenClassification
import tensorflow as tf
import transformers
from transformers import BertForTokenClassification, AdamW
import Seq2SeqUtility as s2s
import numpy as np
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, accuracy_score
import pickle

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForTokenClassification.from_pretrained('roberta-base')

MAX_LEN = 12
BATCH_SIZE = 32



def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def get_ons_accuracy(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    total = [1 if preds[i] == y[i] and y[i] == 1 else 0 for i in range(len(y))]
    all_wild = [1 if y[i] == 1 else 0 for i in range(len(y))]
    return sum(total) / sum(all_wild)

def get_false_negatives(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    total = [1 if preds[i] != y[i] and y[i] == 1 else 0 for i in range(len(y))]
    all_wild = [1 if y[i] == 1 else 0 for i in range(len(y))]
    return sum(total) / sum(all_wild)

def get_false_positives(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    total = [1 if preds[i] != y[i] and y[i] == 0 else 0 for i in range(len(y))]
    all_wild = [1 if y[i] == 0 else 0 for i in range(len(y))]
    return sum(total) / sum(all_wild)

def prepare_examples_for_torch(examples):
    """

    :param exmpples:
    :return:
    """
    examples_sents = []
    examples_labels = []
    for example in examples:
        sent, labels  = tokenize_and_preserve_labels(example.get_sent_words(), example.get_tags())
        examples_sents.append(sent)
        examples_labels.append(labels)
    print(examples_sents)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in examples_sents],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")
    tags = pad_sequences([[l for l in lab] for lab in examples_labels],
                         maxlen=MAX_LEN, value=-1.0, padding="post",
                         dtype="long", truncating="post")
    attention_mask = [[float(i != 0.0) for i in ii] for ii in input_ids]

    return torch.tensor(input_ids).type(torch.LongTensor), torch.tensor(tags).type(torch.LongTensor), torch.tensor(attention_mask)


def test_model(lr, wd,eps):


    snowclone_db_path = "patterns_db_test"
    w2v_path = ""
    sp_reader = s2s.SentencePatternReader(snowclone_db_path)
    # w2v = s2s.get_w2v("snowclone_w2v.pkl", sp_reader, should_create=False)
    # print(w2v["START_WORD_STR"])
    # getting train-val-test split
    train_perc = 0.7
    val_perc = 0.15
    train_examples,val_examples,test_examples = sp_reader.get_train_val_test_examples(train_perc,val_perc)
    tr_inputs, tr_tags, tr_masks = prepare_examples_for_torch(train_examples)
    val_inputs, val_tags, val_masks = prepare_examples_for_torch(val_examples)
    test_inputs, test_tags, test_masks = prepare_examples_for_torch(test_examples)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        # num_labels=3,
        output_attentions = False,
        output_hidden_states = False
    )
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': wd},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=eps
    )

    epochs = 5
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if l_i != -1]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if l_i != -1]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        ons_sum = np.sum([1 if pred_tags[i]==1 else 0 for i in range(len(pred_tags))])
        if (ons_sum >0):
            x = 1
        print("Ons accuracy is: {}".format(get_ons_accuracy(pred_tags,valid_tags)))
        print("False positives: {}".format(get_false_positives(pred_tags,valid_tags)))
        print("False negatives: {}".format(get_false_negatives(pred_tags,valid_tags)))
        # print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print("the sum of ons is " + str(ons_sum))
        return (accuracy_score(pred_tags, valid_tags), get_ons_accuracy(pred_tags,valid_tags), model)

def example_to_snowclone_pattern(example,model):
    #preparing the exmaples:
    inputs,labels,mask = prepare_examples_for_torch([example])
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have not provided labels.
        outputs = model(inputs, token_type_ids=None,
                        attention_mask=mask, labels=labels)
    # Move logits and labels to CPU
    logits = outputs[1].detach().cpu().numpy()
    cur_tags = [list(p) for p in np.argmax(logits, axis=2)]
    new_sent_words = []
    orig_sent_words = example.get_sent_words()
    for i in range(min(len(orig_sent_words),12)):
        try:
            if cur_tags[0][i] == 1:
                new_sent_words.append("*")
            else:
                new_sent_words.append(orig_sent_words[i])
        except:
            stop = 1
    return new_sent_words


def cross_validate(fname):
    model_to_acc = {}
    eps = [0.001,0.00001]
    lr = [0.001,0.0001]
    wd = [0,0.001]
    # eps = [0.1]
    # lr =[0.1]
    # wd = [0.1]
    with open (fname,"w") as f:
        for i in range(10):
            for e in eps:
                for l in lr:
                    for w in wd:
                        if ((e,l,w) not in model_to_acc):
                            model_to_acc[(e,l,w)] = []
                        val_acc, ons_acc,_ = test_model(l,w,e)
                        model_to_acc[(e,l,w)].append((val_acc,ons_acc))
        for model in model_to_acc:
            cur_val_acc = [tup[0] for tup in model_to_acc[model]]
            cur_ons_acc = [tup[1] for tup in model_to_acc[model]]
            f.write("----------------\n")
            f.write("results for model " + str(model) + "\n")
            f.write("validation accuracy is " + str(sum(cur_val_acc) / len(cur_val_acc))+ "\n")
            f.write("validation ones accuracy is " + str(sum(cur_ons_acc) / len(cur_ons_acc))+ "\n")
            f.write("-----------------------------------"+ "\n")

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_best_model(eps,lr,wd,save_name):
    val_acc, ons_acc,model = test_model(lr, wd, eps)
    save_pickle(model,save_name)
    print(val_acc,ons_acc)


#
tag_values = [0,1,-1]
eps = 1e-5
lr = 0.0001
wd = 0
save_name = ("roberta_s2s_1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
# # # sent = "beam me up scotty"
# # # model = load_pickle("nirnirnir")
# # # example = s2s.PhraseExample(sent,[])
# # # print(example_to_snowclone_pattern(example,model))
# #
# cross_validate("nir_roberta_s2s2")
get_best_model(eps,lr,wd,save_name)