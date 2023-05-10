# 预处理原始数据
python preprocess.py ./raw/ ./w2v/GoogleNews-vectors-negative300.bin
# 提取规则特征
python features.py ./stsa.binary.p 