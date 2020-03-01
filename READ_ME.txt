Prerequisite:
==============
keras
numpy
pickle

For train script:
=================
python train.py --train_file train.txt

For test script:
================
python test.py --input_file test.txt --output_file result.txt

Files:
LSTM_KR_PoS.py :  Character embedding model for Korean part of speech tagging
train.py : Training script 
test.py: Testing script


For testing with already train model put all of the saved model files in a 
same directory as test.py 

Model File List:
decoder_model_pos_kr_v10000.json
decoder_model_weights_pos_kr_v10000.h5
encoder_input_data_v.data
encoder_model_pos_kr_v10000.json
encoder_model_weights_pos_kr_v10000.h5
input_texts_v.data
input_token_index_v.data
max_decoder_seq_length_v.data
num_decoder_tokens_v.data
s2s_pos_kr_v10000.h5
target_token_index_v.data

Train with valid.txt:
(Due to memory resource constraint in my computer, current saved model trained with a first 10000 data point from valid.txt files
)

Train with train.txt:
"Train_Save_Model_10000" folder contain all learned model using train.txt with a first 10000 data point. 
Model files Names:
decoder_model_pos_kr_10000.json
decoder_model_weights_pos_kr_10000.h5
encoder_input_data.data
encoder_model_pos_kr_10000.json
encoder_model_weights_pos_kr_10000.h5
input_texts.data
input_token_index.data
max_decoder_seq_length.data
num_decoder_tokens.data
s2s_pos_kr_10000.h5
target_token_index.data



