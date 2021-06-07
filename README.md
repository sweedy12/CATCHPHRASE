# CATCHPHRASE

<h2> Overall Information </h2> </br>
This repository contains the code, web-extension and datasets for the work done in [enter  arxiv url].

The chrome_extension directory contains the code and supplementaries for the CATCHPHRASE web-extension, capable of detecting and highlighting pop-culture references in real-time.

The snowclone_form_tagger directory contains all code used to train, validate and test both BI-LSTM-CRF and BERT models for finding the underlying pattern of an input sentence. It also contains the dataset used for this task. 
The snowclone_classification directory holds the code used for the training process of SVM and RoBERTa models for the task of detecting snowclone usages. The dataset for this task can also be found under this directory.





<h2> CATCHPHRASE Chrome Extension </h2>
The chrome_extension directory holds all code and supplementary matterial needed to run the CATCHPHRASE extension.
To install the extension on your browser, simply upload the entire folder as an extension.


<h2>Snowclone Form Tagger </h2>
<h3> Dataset </h3>
The dataset is we curated for this task consists 7700 pairs of (snowclone pattern, pattern realization), creating pairs of (sentence, labels) where labels[i] = 0 if i'th word may not be used. </br>
The data for this task is found at snowclone_form_tagger/snowclone_form_data. Each row contains 3 sentences separated by commas : </br>
    1. The snowclone reference. </br>
    2. The snowclone reference, where wildcard words have been replaced by "*". </br>
    3. The original snowclone pattern. </br> </br>
<h3> Code </h3> </br>
The entire code for the snowclone form tagger is found under snowclone_form_tagger. To train, validate and test each of the suggested models: </br>
<h5> BI-LSTM-CRF :  </h5> To cross validate and save the best BI-LSTM-CRF model for this task, simply run snowclone_form_tagger/BI_LSTM_CRF_CV.py. The best model, along with a log file containing all the results, will be saved.
<h5> BERT : </h5> To fine tune BERT for this task, simply run bert_snowclone.py

<h2>Snowclone Detection as Binary Classification </h2>
<h3> Dataset </h3>
The dataset curated for this task consists of 3850 pairs of (seed,candidate) pairs, each labeled as reference/non-reference (1/0).
The entire dataset can be found at snowclone_classification/snowclone_reference_data. </br>
The code for training and validating both models can be run the following way: </br>
<h5> SVM :  </h5> for training, validating and saving the best feature-based SVM model, simply run SVMFinder.py
<h5> RoBERTa : </h5> for training, validating and saving the best RoBERTa model for this task, simply run roberta_snowclone.py


    
 
