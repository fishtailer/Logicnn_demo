import torch

import numpy as np


class FOL:
    # 一阶逻辑规则

    def __init__(self, K, input, fea):
        """
        K:种类数量
        """
        self.input = input
        self.fea = fea

        self.conds = self.conditions(self.input, self.fea)  # 记录数据相关性
        self.K = K

    def conditions(self, X, F):
        """输入数据和特征数据，计算相关性(F感觉也不一定是特征数据，但是conds的结果是判断输入数据x是否满足规则)"""
        result = []
        for x, f in zip(X, F):
            result.append(self.condition_single(x, f))  # 将结果加入列表
        # print(torch.stack(result))
        return torch.stack(result)

    def distribution_helper_helper(self, x, f):
        """计算某个样本在所有类别上的条件概率"""
        result = []
        for k in range(self.K):
            result.append(self.value_single(x, k, f))
        #print(result)
        return torch.stack(result)

    def distribution_helper(self, w, X, F, conds):
        """计算每个样本在所有类别上的条件概率
            def distribution_helper(self, w, X, F, conds):
                nx = X.shape[0]
                distr = T.alloc(1.0, nx, self.K)    # 所有初始概率都设为1
                distr,_ = theano.scan(
                    lambda c,x,f,d: ifelse(T.eq(c,1.), self.distribution_helper_helper(x,f), d),
                    sequences=[conds, X, F, distr])
                # 将当前数据x,特征数据f和上一步的distr传入lambda函数中
                lambda函数首先判断当前条件是否满足，即T.eq(c,1)，若满足则去调用helper_helper，计算类别的条件概率并存入distr
                                                                 若不满足则不变，放入distr中
                distr,_ = theano.scan(
                    lambda d: -w*(T.min(d,keepdims=True)-d), # relative value w.r.t the minimum
                    sequences=distr)
                # 计算distr中每个值与最小值的差，并乘上权重w，以使得最后的结果在[0,1]
            return distr
        """
        nx = X.shape[0]
        distr = torch.ones((nx, self.K), dtype=torch.float32)  # 生成size = (数据数量，数据类别)的张量，初始化为1（默认符合逻辑规则）
        for i in range(nx):
            for k in range(self.K):
                if conds[i] == 1.:
                # 如果某个数据符合条件
                    distr[i, k] = self.distribution_helper_helper(X[i], F[i])[k]  # 计算每个符合条件的样本属于某一类的概率
        distr = -w * (torch.min(distr, dim=1, keepdims=True)[0] - distr)  # torch.min返回最小值及其索引

        return distr

    """
    Interface function of logic constraints
    The interface is general---only need to overload condition_single(.) and
    value_single(.) below to implement a logic rule---but can be slow
    See the overloaded log_distribution(.) of the BUT-rule for an efficient
    version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, F=None, config={}):
        """
        return an nxk matrix with the (i,c)-th term
    = - w * (1 - r(X_i, y_i = c))
        if X_i is a grouding of the rule
    = 1 otherwise
        计算每个样本在所有类别上的条件概率，并返回一个矩阵，其中矩阵中的每个元素表示对应样本在对应类别上的条件概率
        """
        if F == None:
            X, F, conds = self.input, self.fea, self.conds  # 如果没有指定特征数据，则用默认的输入和特征
        else:
            conds = self.conditions(X, F)
        log_distr = self.distribution_helper(w, X, F, conds)
        return log_distr

    def condition_single(self, x, f):
        """
        判断样本是否满足条件
        """
        #return (x.mean() + f.mean())/2
        return torch.tensor(0, dtype=torch.float32)
	

    def value_single(self, x, y, f):
        """
        value = r(x,y)，计算样本在某个类别上的条件概率
        """
        #return torch.dot(x,f)*torch.tensor(y, dtype=torch.float32)
        return torch.tensor(1, dtype=torch.float32)

class FOL_BUT(FOL):
    """x = x1_but_x2 => {y => pred(x2) and pred(x2) => y}"""

    def __init__(self, k, input, fea):
        """
        type k:int
        param k:the number of classes

        type fea:
        param fea:symbolic feature tensor, of shape 3
                  fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                  fea[1:2] : classifier.predict_p(x_2)
        """
        assert k == 2
        super(FOL_BUT, self).__init__(k, input, fea)

    """
    rule - specific funtions
    """

    def condition_single(self, x, f):
        return torch.eq(f[0], 1.0).float()

    def value_single(self, x, y, f):
        ret = torch.mean(torch.min[1. - y + f[2], 1.], torch.min[1. - f[2] + y, 1.])
        ret = ret.float()
        """
        if self.condition_single == 1.:
            return ret
        else:
            return torch.tensor(1.).float()
        """
        return torch.where(torch.eq(self.condition_single(x, f), 1.), ret, torch.tensor(1.).float())

    """
    Efficient version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, F=None):
        if F == None:
            x, F = self.input, self.fea
        F_mask = F[:, 0]  # 是否满足But结构
        F_fea = F[:, 1:]  # classifier.predict_p(x_2)
        # y = 0
        distr_y0 = w * F_mask * F_fea[:, 0]
        # y = 1
        distr_y1 = w * F_mask * F_fea[:, 1]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        distr_y1 = distr_y1.reshape([distr_y1.shape[0], 1])
        distr = torch.cat([distr_y0,distr_y1], dim = 1)
        return distr
        

