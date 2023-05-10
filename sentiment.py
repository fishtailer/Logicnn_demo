import pickle,os,sys
import numpy as np
import torch
from collections import defaultdict, OrderedDict
from tqdm import tqdm

def Iden(x):
    y = x
    return(y)


def train_conv_net(datasets,
                   U,  # U为word2vec
                   word_idx_map,
                   img_w=300,
                   filter_hs=[3, 4, 5],
                   hidden_units=[100, 2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=11,
                   batch_size=50,
                   lr_decay=0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   pi_params=[1., 0],
                   C=1.0,
                   patience=20):
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0]) - 1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
    parameters = [("image shape", img_h, img_w), ("filter shape", filter_shapes), ("hidden units", hidden_units),
                  ("dropout", dropout_rate), ("batch_size", batch_size), ("non_static", non_static),
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear), ("sqr_norm_lim", sqr_norm_lim),
                  ("shuffle_batch", shuffle_batch), ("pi_params", pi_params), ("C", C)]
    print(parameters)

    index = torch.tensor([], dtype=torch.long)
    x = torch.tensor([], dtype=torch.float32)
    y = torch.tensor([], dtype=torch.long)
    Words = torch.tensor(U)
    zero_vec_tensor = torch.tensor([], dtype=torch.float32)  ######
    zero_vec = np.zeros(img_w)
    set_zero = lambda zero_vector_tensor: Words[0].copy_(torch.from_numpy(zero_vec_tensor))  # 第一行复制为z_v_t
    layer0_input = Words[x.long().flatten()].view(x.shape[0], 1, x.shape[1], Words.shape[1])
    conv_layers = []
    layer1_inputs = []
    for index in range(len(filter_hs)):
        filter_shape = filter_shapes[index]
        pool_size = pool_sizes[index]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, img_h, img_w),
                                        filter_shape=filter_shape, pool_size=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)  # output输出一个(bs,output_channel, height, width)，因此打平到第二维
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = torch.cat(layer1_inputs, dim=1)
    hidden_units[0] = feature_maps * len(filter_hs)
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations,
                            dropout_rates=dropout_rate)

    # BUT-RULE feature
    f_but = torch.tensor([], dtype=torch.float32, requires_grad=True)
    f_but_ind = torch.tensor([], dtype=torch.float32, requires_grad=True)
    f_but_layer0_input = Words[f_but.long().flatten()].view(f_but.shape[0], 1, f_but.shape[1], Words.shape[1])
    f_but_pred_layers = []
    for conv_layer in conv_layers:
        f_but_layer0_output = conv_layer.predict(f_but_layer0_input, batch_size)  # 卷积之后的结果
        f_but_pred_layers.append(f_but_layer0_output.flatten(2))
    f_but_layer1_input = torch.cat(f_but_pred_layers, dim=1)
    f_but_y_pred_p = classifier.predict_p(f_but_layer1_input)  # softmax的输出
    f_but_full = torch.cat([f_but_ind, f_but_y_pred_p], dim=1)  # but_ind用于检测这句话有没有but,后面那个是这句话的softmax输出
    f_but_full = f_but_full.requires_grad_(False)

    # add logic layer
    nclasses = 2
    rules = [FOL_But(nclasses, x, f_but_full)]
    rules = [FOL_But(nclasses, x, f_but_full)]
    rule_lambda = [1]
    new_pi = get_pi(cur_iter=0, params=pi_params)
    logic_nn = LogicNN(rng, input=x, network=classifier, rules=rules, rule_lambda=rule_lambda, pi=new_pi, C=C)

    # parameters and update
    params_p = logic_nn.params_p
    for conv_layer in conv_layers:
        params_p += conv_layer.params
    if non_static:
        # 如果词向量允许改变，则把他们加入到模型参数
        params_p += [Words]
    cost_p = logic_nn.negative_log_likelihood(y)
    dropout_cost_p = logic_nn.dropout_negative_log_likelihood(y)
    grad_updates_p = sgd_updates_adadelta(params_p, dropout_cost_p, lr_decay, 1e-6, sqr_norm_lim)

    # shuffle dataset and assign to mini batches
    np.random.seed(3435)
    # train data
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        # shuffle both train data and features
        permutation_order = np.random.permutation(datasets[0].shape[0])
        train_set = datasets[0][permutation_order]
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
        new_fea = {}
        train_fea = datasets[3]
        for k in train_fea.keys():
            train_fea_k = train_fea[k][permutation_order]
            extra_fea = train_fea_k[:extra_data_num]
            new_fea[k] = np.append(train_fea[k], extra_fea, axis=0)
        train_text = datasets[6][permutation_order]
        extra_text = train_text[:extra_data_num]
        new_text = np.append(datasets[6], extra_text, axis=0)
    else:
        new_data = datasets[0]
        new_fea = datasets[3]
        new_text = datasets[6]
    # shuffle both training data and features
    permutation_order = np.random.permutaiton(new_data.shape[0])
    new_data = new_data[permutation_order]
    for k in new_fea.keys():
        new_fea[k] = new_fea[k][permutation_order]
    new_text = new_text[permutation_order]
    n_batches = new_data.shape[0] / batch_size
    n_train_batches = n_batches
    train_set = new_data
    train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:, -1]))
    train_fea = new_fea
    train_fea_but_ind = train_fea["but_ind"].reshape([train_fea["but_ind"].shape[0], 1])
    train_fea_but_ind = shared_fea(train_fea_but_ind)
    for k in new_fea.keys():
        if k != "but_text":
            train_fea[k] = shared_fea(new_fea[k])
    # val data
    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[1].shape[0] % batch_size
        # shuffle both val data and features
        permutation_order = np.random.permutation(datasets[1].shape[0])
        val_set = datasets[1][permutation_order]
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[1], extra_data, axis=0)
        new_val_fea = {}
        val_fea = datasets[4]
        for k in val_fea.keys():
            val_fea_k = val_fea[k][permutation_order]
            extra_fea = val_fea_k[:extra_data_num]
            new_val_fea[k] = np.append(val_fea[k], extra_fea, axis=0)
        val_text = datasets[7][permutation_order]
        extra_text = val_text[:extra_data_num]
        new_val_text = np.append(datasets[7], extra_text, axis=0)
    else:
        new_val_data = datasets[1]
        new_val_fea = datasets[4]
        new_val_text = datasets[7]

    val_set = new_val_data
    val_set_x, val_set_y = shared_dataset((val_set[:, :img_h], val_set[:, -1]))
    n_batches = new_val_data.shape[0] / batch_size
    n_val_batches = n_batches
    val_fea = new_val_fea
    val_fea_but_ind = val_fea["but_ind"].reshape([val_fea["but_ind"].shape[0], 1])
    val_fea_but_ind = shared_fea(val_fea_but_ind)
    for k in val_fea.keys():
        if k != "but_text":
            val_fea[k] = shared_fea(val_fea[k])

            # test data
    test_set_x = datasets[2][:, :img_h]
    test_set_y = np.asarray(datasets[2][:, -1], "int32")
    test_fea = datasets[5]
    test_fea_but_ind = test_fea['but_ind']
    test_fea_but_ind = test_fea_but_ind.reshape([test_fea_but_ind.shape[0], 1])
    test_text = datasets[8]

    ######


    # setup testing
    test_size = test_set_x.shape[0]
    test_pred_layers = []
    test_layer0_input = Words[x.long().flatten()].view(test_size, 1, img_h, Words.shape[1])
    f_but_test_pred_layers = []
    f_but_test_layer0_input = Words[f_but.long().flatten()].view(test_size, 1, img_h, Words.shape[1])
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        f_but_test_layer0_output = conv_layer.predict(f_but_test_layer0_input, test_size)
        f_but_test_pred_layers.append(f_but_test_layer0_output.flatten(2))
    test_layer1_input = torch.cat(f_but_test_pred_layers, dim=1)
    f_but_test_layer1_input = torch.cat(f_but_test_pred_layers, dim=1)
    f_but_test_y_pred_p = classifier.predict_p(f_but_test_layer1_input)
    f_but_test_full = torch.cat([f_but_ind, f_but_test_y_pred_p], dim=1)

    test_set_x_shr, test_set_y_shr = shared_dataset((test_set_x, test_set_y))

    test_q_y_pred, test_p_y_pred = logic_nn.predict(test_layer1_input,
                                                    test_set_x_shr,
                                                    [f_but_test_full])
    test_q_error = torch.mean(torch.ne(test_q_y_pred, y))
    test_p_error = torch.mean(torch.ne(test_p_y_pred, y))
    test_model_all =  ####

    # start training over mini_batches
    print("...training")
    batch = 0
    best_val_q_perf = 0
    val_p_perf = 0
    val_q_perf = 0
    cost_epoch = 0
    stop_count = 0
    for index in tqdm(range(n_epochs)):
        # train
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                batch += 1
                new_pi = get_pi(cur_iter=batch * 1. / n_train_batches, params=pi_params)
                logic_nn.set_pi(new_pi)

                dropout_cost_epoch = logic_nn.dropout_negative_log_likelihood(y)
                sgd_updates_adadelta(params_p,dropout_cost_epoch, lr_decay, 1e-6, sqr_norm_lim)

                set_zero(zero_vec)
        else:
            for minibatch_index in range(n_train_batches):
                batch += 1
                new_pi = get_pi(cur_iter=batch * 1. / n_train_batches, params=pi_params)
                logic_nn.set_pi(new_pi)
                cost_epoch = train_model(minibatch_index)
                # eval
        ########
        for index in range(n_train_batches):
            x = train_set_x[index * batch_size: (index + 1) * batch_size]
            y = train_set_x[index * batch_size: (index + 1) * batch_size]
            f_but = train_fea["but"][index * batch_size: (index + 1) * batch_size]
            f_but_ind = train_fea_but_ind[index * batch_size: (index + 1) * batch_size]

            f_but_layer0_input = Words[f_but.long().flatten()].view(f_but.shape[0], 1, f_but.shape[1], Words.shape[1])
            f_but_pred_layers = []
            for conv_layer in conv_layers:
                f_but_layer0_output = conv_layer.predict(f_but_layer0_input, batch_size)  # 卷积之后的结果
                f_but_pred_layers.append(f_but_layer0_output.flatten(2))
            f_but_layer1_input = torch.cat(f_but_pred_layers, dim=1)
            f_but_y_pred_p = classifier.predict_p(f_but_layer1_input)  # softmax的输出
            f_but_full = torch.cat([f_but_ind, f_but_y_pred_p], dim=1)  # but_ind用于检测这句话有没有but,后面那个是这句话的softmax输出
            f_but_full = f_but_full.requires_grad_(False)
            # add logic layer
            nclasses = 2
            rules = [FOL_But(nclasses, x, f_but_full)]
            rule_lambda = [1]
            new_pi = get_pi(cur_iter=0, params=pi_params)
            logic_nn = LogicNN(rng, input=x, network=classifier, rules=rules, rule_lambda=rule_lambda, pi=new_pi, C=C)

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = torch.tensor(data_x, dtype=torch.float32)
    shared_y = torch.tensor(data_y, dtype=torch.int32)
    if torch.cuda.is_available():
        shared_x = shared_x.cuda()
        shared_y = shared_y.cuda()
    return shared_x, shared_y

def shared_fea(fea):
    """
    Function that loads the features into shared variables
    """
    shared_fea = torch.tensor(fea, dtype=torch.float32)
    if torch.cuda.is_available():
        shared_fea = shared_fea.cuda()
    return shared_fea

def get_pi(cur_iter, params = None, pi = None):
    """pi_t = max{1 - k^t, lb}"""
    k,lb = params[0],params[1]
    pi = 1. - max([k**cur_iter, lb])
    return pi


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name="Words"):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        # empty = np.zeros_like(param.get_value()) # 创建一个形状为参数的全0矩阵
        empty = np.zeros_like(param.data)
        exp_sqr_grads[param] = nn.Parameter(as_float(empty), requires_grad=False)
        gp = torch.autograd.grad(cost, param)[0]  # 求param对cost的导
        exp_sqr_ups[param] = nn.Parameter(as_float(empty), requires_grad=False)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * torch.square(gp)  # 二阶动量
        updates[exp_sg] = up_exp_sg
        step = -(torch.sqrt(exp_su + epsilon) / torch.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * torch.sqrt(step)
        if param.name == "Words":
            stepped_param = param + step * .3
        else:
            stepped_param = param + step
        if (param.data.ndimension() == 2) and (param.name != "Words"):
            col_norms = torch.sqrt(torch.sum(torch.square(stepped_param), dim=0))
            desired_norms = torch.clamp(col_norms, 0, torch.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def as_float(variable):
    if isinstance(variable, float):
        return torch.tensor(variable).to(torch.float)
    if isinstance(variable, np.ndarray):
        return torch.tensor(variable).to(torch.float)
    return torch.tensor(variable).to(torch.float)

def get_idx_from_sent(sent, word_idx_map, max_l = 51, k = 300, filter_h = 5):
    """
    Transforms sentence into a list of indices. Pad with zeros
    e.g. s="你好啊"=> x= [0,0,a_1,a_2,a_3,0,0,0,0,]
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)        # 要在前后分别填充filter-1
    return x

# get"but"-rule feature
def get_idx_from_but_fea(but_fea, but_ind, word_idx_map, max_l = 51, k = 300, filter_h = 5):
    if but_ind == 0:
        pad = filter_h - 1
        x = [0] * (max_l + 2 * pad) # 没有but存在，就全为0
    else:
        x = get_idx_from_sent(but_fea, word_idx_map, max_l, k, filter_h)# 只看but的文本
    return x

def make_idx_data(revs, fea, word_idx_map, max_l = 53, k = 300, filter_h = 5):
    """
    Transforms sentences into a 2-d matirx
    but为文本内容索引，but_text为but之后的文本内容，but_ind表示这句话是是否存在"but",存在为1
    """
    train, dev, test = [], [], []
    train_text, dev_text, test_text = [], [], []
    train_fea, dev_fea, test_fea = {},{},{}
    fea["but"] = []
    for k in fea.keys():
        # fea有三个Keys，分别为text,ind和but
        train_fea[k], dev_fea[k], test_fea[k] = [], [], []
    for i,rev in enumerate(revs):
        sent = get_idx_from_sent(rev["text"],word_idx_map, max_l, k, filter_h) # 得到初始文本的索引
        sent.append(rev["y"])
        fea["but"].append(get_idx_from_but_fea(fea["but_text"][i],fea["but_ind"][i],word_idx_map,max_l,k,filter_h)) # 得到but文本的索引
        if rev["split"] == 0:
            train.append(sent)
            for k,v in fea.items():
                # .items()返回字典中所有的元组(key,value)
                train_fea[k].append(v[i]) # 当前这句话中每个key所对应的value，故train_fea为字典，内有三个元组，分别为{but:..,but_text:...,but_ind:...}
            train_text.append(rev["text"])
        elif rev["split"] == 1:
            dev.append(sent)
            for k,v in fea.items():
                dev_fea[k].append(v[i]) # 当前这句话中每个key所对应的value
            dev_text.append(rev["text"])
        else:
            test.append(sent)
            for k,v in fea.items():
                test_fea[k].append(v[i]) # 当前这句话中每个key所对应的value
            test_text.append(rev["text"])
    train = np.array(train, dtype = "int")
    dev = np.array(dev, dtype = "int")
    test = np.array(test, dtype = "int")
    for k in fea.keys():
        if k=='but':
            train_fea[k] = np.array(train_fea[k],dtype='int')
            dev_fea[k] = np.array(dev_fea[k],dtype='int')
            test_fea[k] = np.array(test_fea[k],dtype='int')
        elif k=='but_text':
            train_fea[k] = np.array(train_fea[k])
            dev_fea[k] = np.array(dev_fea[k])
            test_fea[k] = np.array(test_fea[k])
        else:
            train_fea[k] = np.array(train_fea[k],dtype=np.float32)
            dev_fea[k] = np.array(dev_fea[k],dtype=np.float32)
            test_fea[k] = np.array(test_fea[k],dtype=np.float32)
    train_text = np.array(train_text)
    dev_text = np.array(dev_text)
    test_text = np.array(test_text)
    # 返回文本所对应的索引，特征，文本内容
    return [train, dev, test, train_fea, dev_fea, test_fea, train_text, dev_text, test_text]


if __name__ == "__main__":
    path = "data/"
    print("data_path:", path)
    print("loading data...")
    x = pickle.load(open(os.path.join(path + "/stsa.binary.p"), "rb"))
    # revs为数据及其标签等，vocab为独热编码词典，W是出现在预训练模型中的w2v,w_i_m为其索引,W2为未出现在预训练的w2v
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")
    print("load features...")
    fea = pickle.load(open(os.path.join(path + "/stsa.binary.p.fea.p"), "rb"))
    # fea包含了fea_ind和fea_text

    print("features loaded!")

    mode = sys.argv[1]
    word_vectors = sys.argv[2]

    if mode == "-nonstatic":
        print("model architecture:CNN-non-static")
        non_static = True
    elif mode == "static":
        print("model architecture:CNN-static")
        non_static = False
    if word_vectors == "-rand":
        print("using:random vectors")
        U = W2
    elif word_vectors == "-word2vec":
        print("using:word2vec vectors,dim = %d" % W.shape[1])
        U = W

    # execfile("classes.py") # 执行该文件
    # q:teacher network; p: student network
    q_results = []
    p_results = []
    datasets = make_idx_data(revs, fea, word_idx_map, max_l=53, k=300, filter_h=5)
    # print(datasets)