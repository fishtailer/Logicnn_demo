import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def ReLU(x):
    return torch.relu(x)
def Sigmoid(x):
    return torch.sigmoid(x)
def Tanh(x):
    return torch.tanh(x)
def Iden(x):
    y = x
    return(y)


class LogisticRegression:
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression
    :type input: torch.tensor
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie

    """
        # 初始化权重矩阵W（size = (n_in,n_out))为全0
        if W is None:
            self.W = nn.Parameter(torch.zeros((n_in, n_out), dtype=torch.float32))
        else:
            self.W = W
        # 初始化偏置b为长度n_out的向量0
        if b is None:
            self.b = nn.Parameter(torch.zeros((n_out,), dtype=torch.float32))
        else:
            self.b = b
        # 计算分布概率
        self.p_y_given_x = F.softmax((torch.matmul(input, self.W) + self.b), dim=1)  # 应该不用改成nn.softmax?
        # print(self.p_y_given_x)
        # 判断属于哪一类
        self.y_pred = torch.argmax(self.p_y_given_x, dim=1)
        # print(self.y_pred)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
    .. math::

    \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
    \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
    \ell (\theta=\{W,b\}, \mathcal{D})

    :type y: torch.tensor
    :param y: corresponds to a vector that gives for each example the
    correct label

    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # 对每一个估计做对数后求平均,即负对数似然
        return -torch.mean(torch.log(self.p_y_given_x)[torch.arange(y.shape[0]), y])
        # retuern F.nll_loss(torch.log(self.p_y_given_x), y)

    def soft_negative_log_likelihood(self, y):
        """The `soft' version of negative_log_likelihood, where y is a distribution
        over classes rather than a one-hot coding
        y是对x每一类的概率分布，而非独热编码
    :type y: torch.tensor
    :param y: corresponds to a vector that gives for each example the distribution
     over classes. y.shape = [n, K]
    """
        return -torch.mean(torch.sum(torch.log(self.p_y_given_x) * y, dim=1))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch

    :type y: torch.tensor
    :param y: corresponds to a vector that gives for each example the
    correct label
    """
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ("y", target.type, "y_pred", self.y_pred.type))
        if str(y.dtype).startswith("int"):
            # represents a mistake in prediction
            return torch.mean(torch.ne(self.y_pred, y))
        else:
            raise NotImplementedError()  # 想在父类中预留一个方法，使该方法在子类中实现，如果子类中没有对该方法进行重写就被调用，则会报错：NotImplementError


class HiddenLayer:
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):

        self.input = input
        self.activation = activation

        if W is None:
            if activation == "nn.ReLU()":
                # if activation.func_name == "ReLU":
                w_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=np.float32)
            else:
                w_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)),
                                                  size=(n_in, n_out)), dtype=np.float32)
            W = nn.Parameter(torch.tensor(w_values, dtype=torch.float32), requires_grad=True)
        if b is None:
            b_values = torch.zeros((n_out,), dtype=torch.float32)
            b = nn.Parameter(b_values, requires_grad=True)
        self.W = W
        self.b = b

        lin_output = torch.matmul(input, self.W) + b  # 这里可能要根据输入改

        self.output = (lin_output if activation is None else activation(lin_output))

        self.params = [self.W, self.b]

def _dropout_from_layer(rng, layer, p):
    """p is the probabliity of dropping an unit"""
    torch.manual_seed(rng.randint(999999))
    mask = torch.empty(layer.shape, device=layer.device).bernoulli_(1-p) # 用二项分布来设置dropout,输出一个随机0/1的矩阵,p=0时全为1
    output = layer * mask.to(layer.dtype)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, W = None, b = None):
        super(DropoutHiddenLayer,self).__init__(rng=rng, input=input, n_in=n_in, n_out=n_out,
                                                activation=activation, W = W, b = b)
        self.output = _dropout_from_layer(rng, self.output, p = dropout_rate)

class MLPDropout:
    def __init__(self, rng, input, layer_sizes, dropout_rates, activations):
        self.weight_matrix_sizes = list(zip(layer_sizes, layer_sizes[1:]))  # 打包了输入输出
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        # first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                                input=next_dropout_layer_input,
                                                activation=activations[layer_counter],
                                                n_in=n_in, n_out=n_out,
                                                dropout_rate=dropout_rates[layer_counter])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # 重新利用dropout中的参数
            next_layer = HiddenLayer(rng=rng,
                                 input=next_layer_input,
                                 activation=activations[layer_counter],
                                 # 用1-p来缩放矩阵权重W
                                 W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                                 b=next_dropout_layer.b,
                                 n_in=n_in, n_out=n_out)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            # 第一层没有上述操作
            layer_counter += 1

        # set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        output_layer = LogisticRegression(
            input=next_layer_input,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # 计算nll
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # 计算p(y|x)
        self.dropout_p_y_given_x = self.dropout_layers[-1].p_y_given_x
        self.p_y_given_x = self.layers[-1].p_y_given_x

        # 计算soft nll
        self.soft_dropout_negative_log_likelihood = self.dropout_layers[-1].soft_negative_log_likelihood
        self.soft_negative_log_likelihood = self.layers[-1].soft_negative_log_likelihood

        # parameter
        self.params = [param for layer in self.dropout_layers for param in
                       layer.params]  # 先前一个for循环，把得到的Layer放到后一个循环，得到对应的param


    def predict(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](torch.matmul(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = F.softmax((torch.matmul(next_layer_input, layer.W) + layer.b), dim = 1)
        return p_y_given_x

class MLP:
    """Multi-Layer Perceptron Class
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron
       :type rng: numpy.random.RandomState
       :param rng: a random number generator used to initialize weights
       :type input: torch.tensor
       :param input: symbolic variable that describes the input of the
       architecture (one minibatch)
       :type n_in: int
       :param n_in: number of input units, the dimension of the space in
       which the datapoints lie
       :type n_hidden: int
       :param n_hidden: number of hidden units
       :type n_out: int
       :param n_out: number of output units, the dimension of the space in
       which the labels lie
       """
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function

        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                   n_in=n_in, n_out=n_hidden,
                                   activation=F.tanh)  # 激活函数可能要改
    # The logistic regression layer gets as input the hidden units
    # of the hidden layer
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output,
                                                 n_in=n_hidden,
                                                 n_out=n_out)

    # L1 norm ; one regularization option is to enforce L1 norm to
    # be small

    # negative log likelihood of the MLP is given by the negative
    # log likelihood of the output of the model, computed in the
    # logistic regression layer

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
    # the parameters of the model are the parameters of the two layer it is
    # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


class LeNetConvPoolLayer:

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: torch.tensor
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(
            filter_shape[1:])  # num input feature maps * filter height * filter width torch.prod函数计算给定张量的乘积
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        if self.non_linear == "none" or self.non_linear == "relu":
            w_bound = 0.01
        else:
            w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=filter_shape), dtype=np.float32)
        self.W = nn.Parameter(torch.tensor(w_values, dtype=torch.float32), requires_grad=True)

        self.b = nn.Parameter(torch.zeros((filter_shape[0],), dtype=torch.float32))
        conv_out = F.conv2d(input=self.input, weight=self.W, bias=self.b, stride=1, padding=0)
        if self.non_linear == "tanh":
            conv_out_tanh = torch.tanh(conv_out + self.b.view(1, -1, 1, 1))
            self.output = F.max_pool2d(conv_out_tanh, kernel_size=self.poolsize)
        elif self.non_linear == "relu":
            conv_out_relu = F.relu(conv_out + self.b.view(1, -1, 1, 1))
            self.output = F.max_pool2d(conv_out_relu, kernel_size=self.poolsize)
        else:
            pooled_out = F.max_pool2d(conv_out, kernel_size=self.poolsize)
            self.output = pooled_out + self.b.view(1, -1, 1, 1)

        self.params = [self.W, self.b]

    def predict(self, new_data, batch_size):
        """预测新数据"""
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])

        conv_out = F.conv2d(input=new_data, weight=self.W, bias=self.b, stride=1, padding=0)
        if self.non_linear == "tanh":
            conv_out_tanh = torch.tanh(conv_out + self.b.view(1, -1, 1, 1))
            output = F.max_pool2d(conv_out_tanh, kernel_size=self.poolsize)
        elif self.non_linear == "relu":
            conv_out_relu = F.relu(conv_out + self.b.view(1, -1, 1, 1))
            output = F.max_pool2d(conv_out_relu, kernel_size=self.poolsize)
        else:
            pooled_out = F.max_pool2d(conv_out, kernel_size=self.poolsize)
            output = pooled_out + self.b.view(1, -1, 1, 1)
        return output

class Logicnn:
    network: object

    def __init__(self, rng, input, network,
                 rules=[], rule_lambda=[], pi=0., C=1.):
        """
        :type input: torch.tensor
        :param input: symbolic image tensor, of shape image_shape
        """
        self.input = input
        self.network = network
        self.rules = rules
        """
        rl = torch.from_numpy(np.asaary(rule_lambda,dtype = float32))
        self.rule_lambda = nn.Parameter(rl, requires_grad = True)

        self.rule_lambda = nn.Parameter(torch.tensor(rule_lambda, dtype = torch.float32), requires_grad = True)
        self.ones = nn.Parameter(torch.ones(len(rules)) * 1., requires_grad = True)
        self.pi = nn.Parameter(torch.tensor(pi), requires_grad = True)
        """
        self.rule_lambda = torch.tensor(rule_lambda, dtype=torch.float32)  # 包含规则的某个函数?
        self.ones = torch.ones(len(rules)) * 1.
        self.pi = torch.tensor(pi, dtype=torch.float32)
        self.C = C
        #  print(self.input,self.network,self.rules,self.rule_lambda,self,self.ones,self.pi,self.C)
        ## q(y|x)
        dropout_q_y_given_x = self.network.dropout_p_y_given_x * 1.0   # 最后一层的输出经过softmax的结果
        q_y_given_x = self.network.p_y_given_x * 1.0
        # 结合规则限制
        distr = self.calc_rule_constraints()
        q_y_given_x *= distr
        dropout_q_y_given_x *= distr
        # 归一化
        n = self.input.shape[0]
        n_dropout_q_y_given_x = dropout_q_y_given_x / torch.sum(dropout_q_y_given_x, dim=1).reshape((n, 1))
        n_q_y_given_x = q_y_given_x / torch.sum(q_y_given_x, dim=1).reshape((n, 1))
        self.dropout_q_y_give_x = n_dropout_q_y_given_x.requires_grad_(False)
        self.q_y_give_x = n_q_y_given_x.requires_grad_(False)  # 可能需要reshape
        # 计算类别
        self.q_y_pred = torch.argmax(q_y_given_x, dim=1)               # 经过规则限制之后，计算属于哪一类（教师）
        self.p_y_pred = torch.argmax(self.network.p_y_given_x, dim=1)  # 规则限制之前属于哪一类（学生）
        # 整理参数
        self.params_p = self.network.params

    def calc_rule_constraints(self, new_data=None, new_rule_fea=None):
        if new_rule_fea == None:
            new_rule_fea = [None] * len(self.rules)
        distr_all = torch.tensor(0, dtype=torch.float32)
        for i, rule in enumerate(self.rules):
            distr = rule.log_distribution(self.C * self.rule_lambda[i], new_data, new_rule_fea[i]) # distribution输出两列张量，均代表了文中计算公式的输出
            distr_all += distr
        distr_all += distr

        distr_y0 = distr_all[:, 0]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        distr_y0_copies = distr_y0.repeat(1, distr_all.shape[1])  # 复制第一列，得到的矩阵和初始矩阵形状相同
        distr_all -= distr_y0_copies                            # 确保第一个输出类（即索引为 0 的类）的对数分布为 0
        distr_all = torch.max(torch.min(distr_all, 60.), -60.)  # 限制范围
        return torch.exp(distr_all)

    def set_pi(self, new_pi):
        return self.pi.set_value(new_pi)

    def get_pi(self):
        return self.pi.get_value()

    def dropout_negative_log_likelihood(self, y):
        nlld = (1. - self.pi) * self.network.dropout_negative_log_likelihood(y)
        nlld += self.pi * self.network.soft_dropout_negative_log_likelihood(self.dropout_q_y_given_x)
        return nlld

    def negative_log_likelihood(self, y):
        nlld = (1. - self.pi) * self.network.negative_log_likelihood(y)
        nlld += self.pi * self.network.soft_negative_log_likelihood(self.dropout_q_y_given_x)
        return nlld

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ("y", target.type, "y_pred", self.q_y_pred.type))
        if str(y.dtype).startswith("int"):
            # represents a mistake in prediction
            return torch.mean(torch.ne(self.q_y_pred, y)), torch.mean(torch.ne(self.p_y_pred, y))
        else:
            raise NotImplementedError()

    def predict(self, new_data_to_network, new_data, new_rule_fea):
        q_y_pred_p, p_y_pred_p = self.predict_p(new_data_to_network, new_data, new_rule_fea)
        q_y_pred = torch.argmax(q_y_pred_p, dim=1)
        p_y_pred = torch.argmax(p_y_pred_p, dim=1)
        return q_y_pred, p_y_pred

    def predict_p(self, new_data_to_network, new_data, new_rule_fea):
        p_y_pred_p = self.network.predict_p(new_data_to_network)
        q_y_pred_p = p_y_pred_p * self.calc_rule_constraints(new_data=new_data, new_rule_fea=new_rule_fea)
        return q_y_pred_p, p_y_pred_p


class Val_model(Logicnn):
    def __init__(self, rng, input, network, rules=[]):
        super(Val_model, self).__init__()





