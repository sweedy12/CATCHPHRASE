#Code to cross validate the S2S model

import Seq2SeqUtility as s2s
from S2SLSTM import train_lstm_with_w2v, train_encoder_decoder_general
import numpy as np
from S2S_LSTM_CRF import train_lstm_crf_with_w2v

N_EPOCHS = 20

class CrossValidate:
    N_TRIALS = 10
    LR = [0.01, 0.001,0.0001]
    WD = [0.001,0.01,0]
    n_layers_opts = [1,2,3]
    hidden_dim_opts = [64,32,128]
    def __init__(self):
        self.model_to_val_measures = {}
        self.initiate_model_to_val_measures()

    def initiate_model_to_val_measures(self):
        for lr in self.LR:
            for wd in self.WD:
                for n in self.n_layers_opts:
                    for h in self.hidden_dim_opts:
                        # self.model_to_val_measures[(lr,wd,n,h,"LSTM")] = []
                        # self.model_to_val_measures[(lr,wd,n,h,"ENC-DEC")] = []
                        self.model_to_val_measures[(lr,wd,n,h,"CRFLSTM")] = []

    def check_lstm_enc_dec(self,train_data,val_data,test_data,lr, wd,n_layers,hidden_dim):
        # val_measures, test_measures = train_lstm_with_w2v(train_data,val_data,test_data,N_EPOCHS,lr,wd,n_layers,hidden_dim)
        # self.model_to_val_measures[(lr,wd,n_layers,hidden_dim,"LSTM")].append(val_measures)
        # ed_val_measures,ed_test_measures = train_encoder_decoder_general(train_data,val_data,test_data,N_EPOCHS,lr,wd,n_layers,hidden_dim)
        # self.model_to_val_measures[(lr,wd,n_layers,hidden_dim,"ENC-DEC")].append(ed_val_measures)
        ed_val_measures,ed_test_measures = train_lstm_crf_with_w2v(train_data,val_data,test_data,N_EPOCHS,lr,wd,n_layers,hidden_dim)
        self.model_to_val_measures[(lr,wd,n_layers,hidden_dim,"CRFLSTM")].append(ed_val_measures)



    def check_all_models_on_data(self,train_data,val_data,test_data):
        for lr in self.LR:
            for wd in self.WD:
                for n in self.n_layers_opts:
                    for h in self.hidden_dim_opts:
                        print("Starting with a new model " + str((lr,wd,n,h)))
                        self.check_lstm_enc_dec(train_data,val_data,test_data,lr,wd,n,h)

    def check_all_models(self):
        snowclone_db_path = "patterns_db_test"
        w2v_path = ""
        sp_reader = s2s.SentencePatternReader(snowclone_db_path)
        w2v = s2s.get_w2v("snowclone_w2v.pkl", sp_reader, should_create=False)
        # print(w2v["START_WORD_STR"])
        # getting train-val-test split
        train_perc = 0.7
        val_perc = 0.15
        for i in range(self.N_TRIALS):
            print("Starting trial number " + str(i))
            print("\n\n\n")
            sp_reader.train_val_test_split(train_perc, val_perc)
            train, val, test = sp_reader.get_train_val_test_X_y(w2v)
            self.check_all_models_on_data(train,val,test)

    def write_results_to_file(self,results_file):
        with open(results_file,"w") as f:
            for model_params in self.model_to_val_measures:
                loss = [x[0] for x in self.model_to_val_measures[model_params]]
                acc = [x[1] for x in self.model_to_val_measures[model_params]]
                ons_acc = [x[2] for x in self.model_to_val_measures[model_params]]
                all_acc = [x[3] for x in self.model_to_val_measures[model_params]]
                f.write("-------------------------------------")
                f.write("results for " + str(model_params[4]) + " model, with params " + str(model_params[:-1]) + "\n")
                f.write("average loss is: " + str(np.mean(loss)) + "\n")
                f.write("average accuracy is: " + str(np.mean(acc)) + "\n")
                f.write("average wildcard accuracy is: " + str(np.mean(ons_acc)) + "\n")
                f.write("average all-correct-accuracy is: " + str(np.mean(all_acc)) + "\n")
                f.write("\n")

    def get_best_model(self,results_file,ind=1):
        max_acc = 0
        max_acc_params = None
        with open(results_file,"a") as f:
            for key in self.model_to_val_measures:
                rel_acc = [x[ind] for x in self.model_to_val_measures[key]]
                mean_acc  = np.mean(rel_acc)
                if (mean_acc >= max_acc):
                    max_acc = mean_acc
                    max_acc_params = key
            f.write("The best model we have is " + str(max_acc_params) + " with accuracy " + str(max_acc))




results_file = "cv_lstm_crf_ner_nir"
cs = CrossValidate()
cs.check_all_models()
cs.write_results_to_file(results_file)
cs.get_best_model(results_file,1)

