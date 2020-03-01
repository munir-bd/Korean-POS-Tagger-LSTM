import sys, getopt
import LSTM_KR_PoS

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "h:", ["train_file="])
    except getopt.GetoptError:
        print('python train.py --train_file train.txt')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python train.py --train_file train.txt')
            sys.exit()
        if opt == '--train_file':
            inputfile = arg
            if arg == "train.txt":
                print('Input file is ', inputfile)
            else:
                print('Wrong Input File Name')
        else:
            print('python train.py --train_file train.txt')
    if inputfile == "train.txt":
        print("inputfile ", inputfile)
        LSTM_KR_PoS.lstm_kr_pos(inputfile)
    else:
        print('python train.py --train_file train.txt')

if __name__ == "__main__":
    main(sys.argv[1:])

