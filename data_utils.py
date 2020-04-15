import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"'s", " ", string)
    return string.strip().lower()



def PreprocessSentLabel(data_dir):
    lines = []
    with open(data_dir,'r') as f :
        line = f.readline()
        lines.append(line)
        while line != '':
            line = f.readline()
            lines.append(line)

    sent_list = []
    label_list = []
    for token in lines:
        if token == '':
            break
        sent = token.split('","')[2]
        label = int(token.split('","')[0][-1])
        sent_list.append(clean_str(sent))
        label_list.append(label)
    return sent_list, label_list

train_sent_list, train_label_list = PreprocessSentLabel('./data/yelp_review_polarity_csv/yelp_review_polarity_csv/train.csv')
test_sent_list, test_label_list = PreprocessSentLabel('./data/yelp_review_polarity_csv/yelp_review_polarity_csv/test.csv')

def Word2Index(sent_list):
    word2idx = {}
    idx2word = {}
    index = 0
    for sent in sent_list:
        line = sent.split(' ')
        for word in line:
            if word in word2idx:
                continue
            else:
                word2idx[word] = index
                index += 1
    for k,v in word2idx.items():
        idx2word[v]=k
    return word2idx, idx2word

vocab_list = []
vocab_list = vocab_list + train_sent_list + test_sent_list
word2idx, idx2word = Word2Index(vocab_list)


