import sys, getopt

def model_testing_code (input_file, result):
    import numpy as np
    import pickle

    test_file = input_file
    # test_file = "test_1.txt"
    output_file = result
    # output_file = "result_1.txt"

    # with open('input_texts_v.data', 'wb') as filehandle:
    #     pickle.dump(input_texts, filehandle)

    with open('input_texts_v.data', 'rb') as filehandle:
        input_texts_from_file = pickle.load(filehandle)
    # print("input_texts_from_file ", input_texts_from_file)

    # with open('encoder_input_data_v.data', 'wb') as filehandle:
    #     pickle.dump(encoder_input_data, filehandle)

    with open('encoder_input_data_v.data', 'rb') as filehandle:
        encoder_input_data_from_file = pickle.load(filehandle)
    # print("encoder_input_data_from_file ", encoder_input_data_from_file)

    # with open('encoder_input_data_v.data', 'wb') as filehandle:
    #     pickle.dump(encoder_input_data, filehandle)

    with open('encoder_input_data_v.data', 'rb') as filehandle:
        encoder_input_data_from_file = pickle.load(filehandle)
    # print("encoder_input_data_from_file ", encoder_input_data_from_file)

    # with open('num_decoder_tokens_v.data', 'wb') as filehandle:
    #     pickle.dump(num_decoder_tokens, filehandle)

    with open('num_decoder_tokens_v.data', 'rb') as filehandle:
        num_decoder_tokens_from_file = pickle.load(filehandle)
    # print("num_decoder_tokens_from_file ", num_decoder_tokens_from_file)

    # with open('target_token_index_v.data', 'wb') as filehandle:
    #     pickle.dump(target_token_index, filehandle)

    with open('target_token_index_v.data', 'rb') as filehandle:
        target_token_index_from_file = pickle.load(filehandle)
    # print("target_token_index_from_file ", target_token_index_from_file)

    # with open('input_token_index_v.data', 'wb') as filehandle:
    #     pickle.dump(input_token_index, filehandle)

    with open('input_token_index_v.data', 'rb') as filehandle:
        input_token_index_from_file = pickle.load(filehandle)
    # print("input_token_index_from_file ", input_token_index_from_file)

    # with open('max_decoder_seq_length_v.data', 'wb') as filehandle:
    #     pickle.dump(max_decoder_seq_length, filehandle)

    with open('max_decoder_seq_length_v.data', 'rb') as filehandle:
        max_decoder_seq_length_from_file = pickle.load(filehandle)
    # print("max_decoder_seq_length_from_file ", max_decoder_seq_length_from_file)

    from keras.models import model_from_json
    def load_model(model_filename, model_weights_filename):
        with open(model_filename, 'r', encoding='utf8') as f:
            model = model_from_json(f.read())
        model.load_weights(model_weights_filename)
        return model

    encoder_read = load_model('encoder_model_pos_kr_v10000.json', 'encoder_model_weights_pos_kr_v10000.h5')
    decoder_read = load_model('decoder_model_pos_kr_v10000.json', 'decoder_model_weights_pos_kr_v10000.h5')

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index_from_file.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index_from_file.items())

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        # print("input_seq.shape ",input_seq.shape)
        states_value = encoder_read.predict(input_seq)
        # print("states_value ", states_value)

        # Generate empty target sequence of length 1.
        # target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq = np.zeros((1, 1, num_decoder_tokens_from_file))

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index_from_file[' ']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        Kr_Pos = ''
        while not stop_condition:
            output_tokens, h, c = decoder_read.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            Kr_Pos += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(Kr_Pos) > max_decoder_seq_length_from_file):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens_from_file))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return Kr_Pos

    def test_from_text_file(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            lines_test = f.read().split('\n')

        # print("lines_text[0] ", lines_test[0])
        # print("len(lines_test)-1 ", len(lines_test)-1)

        for test_item_ind in range(len(lines_test) - 1):
            # print("test_item_ind ", test_item_ind, " lines_test ", lines_test[test_item_ind])
            print("Input for test: ", lines_test[test_item_ind])

            if input_texts_from_file.count(lines_test[test_item_ind]) > 0:
                index_in_enc = input_texts_from_file.index(lines_test[test_item_ind])
                # print("index_in_enc ", index_in_enc)
                # encode and decode from the model
                # input_seq = encoder_input_data[index_in_enc: index_in_enc + 1]
                input_seq = encoder_input_data_from_file[index_in_enc: index_in_enc + 1]
                Kr_Pos = decode_sequence(input_seq)
                print('KR POS Tag:', Kr_Pos)
                with open(output_file, "a", encoding='utf8') as myfile:
                    myfile.write(Kr_Pos)
                    # myfile.write("\n")
            else:
                print("KR POS Tag: Out of Voc")
                with open(output_file, "a", encoding='utf8') as myfile:
                    myfile.write("Out of Voc")
                    myfile.write("\n")

    test_from_text_file(test_file)

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
       opts, args = getopt.getopt(argv, "h:",["input_file=","output_file="])
    except getopt.GetoptError:
        print('python test.py --input_file test.txt --output_file result.txt')
        sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
           print('python test.py --input_file test.txt --output_file result.txt')
           sys.exit()
       elif opt == '--input_file':
           inputfile = arg
           if arg == "test.txt":
               inputfile = arg
               # print('Input file is ', inputfile)
           else:
               print('Wrong Input File Name')
       elif opt == '--output_file':
           outputfile = arg
           if arg == "result.txt":
               outputfile = arg
               # print('Output file is ', outputfile)
           else:
               print('Wrong Output File Name')
       else:
           print('python test.py --input_file test.txt --output_file result.txt')

    # print("inputfile ", inputfile, "outputfile ", outputfile)
    if inputfile == "test.txt" and outputfile == "result.txt":
        print("inputfile ", inputfile, "outputfile ", outputfile)
        model_testing_code(inputfile, outputfile)
    else:
        print('python test.py --input_file test.txt --output_file result.txt')



if __name__ == "__main__":
   main(sys.argv[1:])


# print("Hello Test")
# print ('train.py --train_file train.txt')