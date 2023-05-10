import os,sys,re
import pickle 
from collections import defaultdict
import numpy as np
from random import randint
import pandas as pd
from tqdm import tqdm


def build_data(data_folder, clean_string=True):
    """
    构建数据集,revs记录了每个样本所对应的标签，文本，文本长度以及具体是哪个数据集
               vocab为构建的词典，采用了ont-hot的形式 比如hello:1，world:2
    """
    revs = []
    [train_file, dev_file, test_file] = data_folder
    vocab = defaultdict(float)
    with open(train_file, "rb") as f:
        for line in f:
            line = line.strip()  # 去除首位空格
            y = int(line[0])  # 首位为标签
            rev = []
            rev.append(line[2:].strip().decode())  # 第二位开始为句首，这里需要decode()解码来转化成字符串
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())  # 以空格为单位，把词分开
            for word in words:
                vocab[word] += 1
            datum = {
                "y": y,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": 0
            }
            revs.append(datum)

    with open(dev_file, "rb") as f:
        for line in f:
            line = line.strip()  # 去除首位空格
            y = int(line[0])  # 首位为标签
            rev = []
            rev.append(line[2:].strip().decode())  # 第二位开始为句首
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())  # 以空格为单位，把词分开
            for word in words:
                vocab[word] += 1
            datum = {
                "y": y,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": 1
            }
            revs.append(datum)

    with open(test_file, "rb") as f:
        for line in f:
            line = line.strip()  # 去除首位空格
            y = int(line[0])  # 首位为标签
            rev = []
            rev.append(line[2:].strip().decode())  # 第二位开始为句首
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())  # 以空格为单位，把词分开
            for word in words:
                vocab[word] += 1
            datum = {
                "y": y,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": 2
            }
            revs.append(datum)
    return revs, vocab

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
    对于在文档中出现了至少一次的词，赋予其在-0.25~0.25之间的任意值
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

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

if __name__ == "__main__":
    stsa_path = sys.argv[1] # 原始数据
    w2v_file = sys.argv[2]  # 预训练模型
    train_data_file = os.path.join(stsa_path + "/stsa.binary.phrases.train")
    dev_data_file = os.path.join(stsa_path + "/stsa.binary.dev")
    test_data_file = os.path.join(stsa_path + "/stsa.binary.test")
    data_folder = [train_data_file, dev_data_file, test_data_file]
    print("loading data...")
    revs, vocab = build_data(data_folder, clean_string = True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])  # 找到最长的句子,pd.dataframe将数据加载成类似表格形式
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec...")
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v,vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs,vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, W2, word_idx_map, vocab], open("./stsa.binary.p","wb")) # 最终保存的预处理数据有revs(包含了每句话标签，文本，长度，train or dev or test),W(词向量矩阵)，以及其所对应的索引，vocab（字典）
    print("dataset created!")
