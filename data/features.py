import pickle
import os, sys

def extract_but(revs):
    but_fea = []    # 记录but之后的文本内容（特征）
    but_ind = []    # 如果某段文本中包含but，则为1，否则为0
    but_fea_cnt = 0 # 统计but出现次数
    for rev in revs:
        text = rev["text"]
        if "but" in text:
            but_ind.append(1)
            # 将but之后的文本作为特征
            fea = text.split("but")[1:] # 由于以but为分割，将文本划分成了两个字符串，因此[1:]开始的文本为特征
            fea = "".join(fea)          # fea中会存储所有的特征
            fea = fea.strip().replace("  "," ") #此处意义不明
            but_fea_cnt += 1
        else:
            but_ind.append(0)
            fea = ''
        but_fea.append(fea)
    print("#but %d" %but_fea_cnt)
    return{"but_text":but_fea, "but_ind":but_ind}


if __name__ == "__main__":
    data_file = sys.argv[1]
    print("loading data...")
    x = pickle.load(open(data_file, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")
    but_fea = extract_but(revs)
    pickle.dump(but_fea, open(os.path.join(data_file + ".fea.p"), "wb"))
    print("features dumped!")
