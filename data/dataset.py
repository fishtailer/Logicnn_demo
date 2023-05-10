import os,re
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, data_path,clean_string = True):
        super(MyDataset, self).__init__()
        self.clean_string = clean_string
        self.revs, self.vocab = self.build_data(data_path)

    def build_data(self, data_path):
        revs = []
        vocab = defaultdict(float)
        with open(data_path, "rb") as f:
            for line in f:
                line = line.strip()
                y = int(line[0])
                rev = []
                rev.append(line[2:].strip().decode())
                if self.clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev)
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum = {
                    "y" : y,
                    "text" : orig_rev,
                    "num_words": len(orig_rev)
                }
                revs.append(datum)
        return revs, vocab

    def __len__(self):
        return len(self.revs)

    def __getitem__(self, idx):
        return self.revs[idx]["text"], self.revs[idx]["y"]

    def get_vocab(self):
        return self.vocab  # 取出字典，在外面合并

def clean_str(string, TREC = False):
    """
    正则匹配所有字符
    除了TREC外的字符全小写
    """
    string = re.sub(r"[^A-Za-z0-9(),!?'\`]"," ",string) # 将除字母和数字 (),!?'\`外的所有字符清除
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
    return string.strip() if TREC else string.strip().lower()


def load_bin_vec(file_name, vocab):
    """
    加载word2vec
    """
    word_vecs = {}
    with open(file_name, "rb") as f:
        header = f.readline()   # 读入文件首行
        vocab_size, layer1_size = map(int, header.split()) # 以空格进行分割，使用map函数将header变量中字符串转换为vocab_size、 layer1_size变量，它们分别用于存储词汇大小和每个单词的向量维度。
        binary_len = np.dtype("float32").itemsize * layer1_size # 计算出每个单词的向量的大小，并将其存储在binary_len变量中。
        for line in tqdm(range(vocab_size)):
            word = []
            while True:
                ch = f.read(1).decode("latin-1") # 读取一个字节的文件，保存在ch中,
                if ch == ' ':
                    word = ''.join(word)
                    break     # 读到空格就是一个单词结束了
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.frombuffer(f.read(binary_len),dtype="float32") # 把字符串转换为float32，然后读取它对对应的embedding
            else:
                f.read(binary_len)  # 不存在就跳过
        return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    对于在文档中至少一次，但不在w2v的词，赋予其在-0.25~0.25之间的任意值
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_W(word_vecs, k=300):
    """
    得到一个词向量矩阵，以及每个单词所对应的索引
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k),dtype = 'float32')
    W[0] = np.zeros(k, dtype = 'float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

#具体调用时用这个，但是vocab好像只构建了每个数据集内部的

'''
train_dataset = MyDataset(train_data_file, clean_string=True)
dev_dataset = MyDataset(dev_data_file, clean_string=True)
test_dataset = MyDataset(test_data_file, clean_string=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
'''
if __name__ =="__main__":
    stsa_path = sys.argv[1] # 原始数据
    w2v_file = sys.argv[2]  # 预训练模型
    train_data_file = os.path.join(stsa_path + "/stsa.binary.phrases.train")
    dev_data_file = os.path.join(stsa_path + "/stsa.binary.dev")
    test_data_file = os.path.join(stsa_path + "/stsa.binary.test")

    # 构建三个dataset
    train_dataset = MyDataset(train_data_file, clean_string=True)
    dev_dataset = MyDataset(dev_data_file, clean_string=True)
    test_dataset = MyDataset(test_data_file, clean_string=True)

    # 合并字典
    vocab = defaultdict(float)
    vocab.update(train_dataset.get_vocab()) # 对于所有出现于数据中的词，都会有记录，合并为字典
    vocab.update(dev_dataset.get_vocab())
    vocab.update(test_dataset.get_vocab())

    w2v = load_bin_vec(w2v_file, vocab) # 得到字典里所有词所对应的vector
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))

    add_unknown_words(w2v,vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs,vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([W, W2, word_idx_map, vocab], open("./stsa.binary1.p", "wb"))





















