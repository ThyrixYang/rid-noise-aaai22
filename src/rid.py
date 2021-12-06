##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.special import softmax

# From FrEIA
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


def get_cINN_model(x_dim, y_dim, hidden_size, layer_num):
    def subnet_fc(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                             nn.Linear(hidden_size,  out_dim))
    cond_node = ConditionNode(y_dim)
    nodes = [InputNode(x_dim, name='input')]
    for i in range(layer_num):
        nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                          {'subnet_constructor': subnet_fc,
                           'clamp': 2.0},
                          conditions=cond_node,
                          name='coupling_{}'.format(i)))
        nodes.append(Node(nodes[-1], PermuteRandom,
                          {'seed': i}, name='permute_{}'.format(i)))
    nodes.append(OutputNode(nodes[-1], name='output'))
    nodes.append(cond_node)
    return ReversibleGraphNet(nodes, verbose=False)


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes, bn, dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.bn = bn
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None):
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out


def train_weights(train_x, train_y, config):
    split_num = 5
    epoch = 2000
    device = config["device"]
    x_dim = train_x.shape[1]
    y_dim = train_y.shape[1]
    train_xs = np.split(train_x, split_num, axis=0)
    train_ys = np.split(train_y, split_num, axis=0)
    index = np.split(
        np.array(list(range(train_x.shape[0]))), split_num, axis=0)
    weights = np.zeros((train_x.shape[0],))
    for sn in range(split_num):
        critic = MLP(x_dim, y_dim, hidden_sizes=[
            256, 256, 256], bn=True, dropout=0.5)
        critic = critic.to(device)
        critic.train()
        optimizer = torch.optim.Adam(critic.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-3)
        schedular = ReduceLROnPlateau(optimizer=optimizer,
                                      factor=0.5,
                                      patience=50,
                                      verbose=True)
        _train_x = []
        _train_y = []
        _train_idx = []
        _vali_idx = None
        for i, (xs, ys, idx) in enumerate(zip(train_xs, train_ys, index)):
            if i == sn:
                _vali_x = xs
                _vali_y = ys
                _vali_idx = idx
            else:
                _train_x.append(xs)
                _train_y.append(ys)
                _train_idx.append(idx)
        _train_x = np.concatenate(_train_x, axis=0)
        _train_y = np.concatenate(_train_y, axis=0)
        _train_idx = np.concatenate(_train_idx, axis=0)
        _vali_x = torch.tensor(_vali_x, dtype=torch.float32, device=device)
        _vali_y = torch.tensor(_vali_y, dtype=torch.float32, device=device)
        _train_x = torch.tensor(_train_x, dtype=torch.float32, device=device)
        _train_y = torch.tensor(_train_y, dtype=torch.float32, device=device)
        best_vali_loss = 1e9
        best_sd = None
        for ep in range(epoch):
            critic.train()
            _pred_y = critic(_train_x)
            loss = torch.mean((_pred_y - _train_y)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedular.step(loss.item())

            critic.eval()
            _pred_y = critic(_vali_x)
            vali_loss = torch.mean((_pred_y - _vali_y)**2)
            if (ep + 1) % 100 == 0 and vali_loss.item() < best_vali_loss:
                best_sd = critic.state_dict()
                best_vali_loss = vali_loss.item()
                print("sn {}, ep {}, train_loss {}, best_vali_loss {}".format(
                    sn, ep, loss.item(), best_vali_loss))
        critic.load_state_dict(best_sd)
        critic.eval()
        _pred_vali_y = critic(_vali_x)
        vali_loss = torch.mean((_pred_vali_y - _vali_y)**2, axis=-
                               1).detach().cpu().numpy()
        weights[_vali_idx] += vali_loss
    weights /= np.mean(weights)
    weights = softmax(-config["tau"]*weights)
    weights /= np.mean(weights)
    weights += 1e-4
    train_n = int(train_x.shape[0]*0.8)
    _train_x, _vali_x = train_x[:train_n], train_x[train_n:]
    _train_y, _vali_y = train_y[:train_n], train_y[train_n:]
    _train_weights, _vali_weights = weights[:train_n], weights[train_n:]
    return {
        "train_x": _train_x,
        "train_y": _train_y,
        "vali_x": _vali_x,
        "vali_y": _vali_y,
        "train_weights": _train_weights,
        "vali_weights": _vali_weights
    }


def train_and_inference(config):
    env = config["env"]
    train_x, train_y, vali_x, vali_y, test_x, test_y = config["train_x"], config[
        "train_y"], config["vali_x"], config["vali_y"], config["test_x"], config["test_y"]
    _train_x = np.concatenate([train_x, vali_x], axis=0)
    _train_y = np.concatenate([train_y, vali_y], axis=0)

    print("------------- training weights ----------------")
    data = train_weights(_train_x, _train_y, config)
    train_x, train_y, train_w, vali_x, vali_y, vali_w =\
        data["train_x"], data["train_y"], data["train_weights"],\
        data["vali_x"], data["vali_y"], data["vali_weights"]

    x_dim = env.input_shape[0]
    y_dim = env.output_shape[0]
    device = config["device"]
    tensor_train_x = torch.tensor(train_x, dtype=torch.float32)
    tensor_train_y = torch.tensor(train_y, dtype=torch.float32)
    tensor_vali_x = torch.tensor(vali_x, device=device, dtype=torch.float32)
    tensor_vali_y = torch.tensor(vali_y, device=device, dtype=torch.float32)
    tensor_vali_w = torch.tensor(vali_w, device=device, dtype=torch.float32)
    train_w = torch.tensor(train_w, dtype=torch.float32).view((-1, 1))
    train_dataset = torch.utils.data.TensorDataset(
        tensor_train_x, tensor_train_y, train_w)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"])
    model = get_cINN_model(x_dim=x_dim, y_dim=y_dim,
                           hidden_size=config["hidden_size"],
                           layer_num=config["layer_num"])
    model = model.to(device)
    model.train()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters,
                                 lr=config["lr"],
                                 betas=config["adam_betas"],
                                 eps=1e-6,
                                 weight_decay=config["weight_decay"])
    lr_schedular = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config["lrs_patience"], threshold=1e-4, verbose=True)
    best_vali_loss = 1e9
    best_sd = None
    best_vali_epoch = None
    for i_epoch in range(config["max_epoch"]):
        model.train()
        for x, y, w in train_loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad()
            z, log_jac_det = model(
                x, c=y)
            ll_loss = torch.sum(z**2, dim=1, keepdim=True) * \
                0.5 - log_jac_det.view((-1, 1))
            wll_loss = w * ll_loss

            loss = torch.mean(wll_loss)

            loss.backward()
            for p in trainable_parameters:
                p.grad.data.clamp_(-15.00, 15.00)
            optimizer.step()

        model.eval()
        vali_z, vali_log_jac_det = model(tensor_vali_x, c=tensor_vali_y)
        ll_loss = torch.sum(vali_z**2, dim=1, keepdim=True) * \
            0.5 - vali_log_jac_det.view((-1, 1))
        wll_loss = tensor_vali_w * ll_loss
        vali_loss = torch.mean(ll_loss)

        lr_schedular.step(vali_loss.item())
        if vali_loss.item() < best_vali_loss:
            best_vali_epoch = i_epoch
            best_vali_loss = vali_loss.item()
            best_sd = model.state_dict()
            print("epoch {}, train_loss {}, vali_loss {}, best_vali_loss {}, best_vali_epoch {}".format(
                i_epoch, loss.item(), vali_loss.item(), best_vali_loss, best_vali_epoch))
    model.load_state_dict(best_sd)

    tensor_test_y = torch.tensor(
        test_y, device=device, dtype=torch.float32)
    model.eval()

    def inference_once():
        tensor_test_z = torch.randn(
            tensor_test_y.shape[0], x_dim, device=device, dtype=torch.float32)
        pred_x, _ = model(tensor_test_z, c=tensor_test_y, rev=True)
        pred_x_np = pred_x.detach().cpu().numpy()
        return pred_x_np

    pred_xs = []
    for _ in range(config["query_num"]):
        pred_xs.append(inference_once())
    pred_xs = np.stack(pred_xs)
    return {
        "test_y": test_y,
        "pred_xs": pred_xs
    }
