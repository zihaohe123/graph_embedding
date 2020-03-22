# -*- coding:utf-8 -*-

import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2

from ..utils import preprocess_nxgraph


def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        b_ = K.ones_like(y_true, dtype=tf.float32)
        # b_[y_true != 0] = beta

        b_ = tf.where(y_true != 0, beta, b_)
        x = K.square((y_true - y_pred) * b_)
        t = K.sum(x, axis=-1)
        return K.mean(t)

    return loss_2nd


def l_1st(alpha):
    def loss_1st(y_true, y_pred):
        L = y_true
        Y = y_pred
        batch_size = tf.cast(K.shape(L)[0], tf.float32)
        return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size

    return loss_1st


def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    A = Input(shape=(node_size,))
    L = Input(shape=(None,))
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2), name='1st')(fc)
        else:
            fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2))(fc)
    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2))(fc)

    A_ = Dense(node_size, 'relu', name='2nd')(fc)
    model = Model(inputs=[A, L], outputs=[A_, Y])
    emb = Model(inputs=A, outputs=Y)
    return model, emb

def create_A_L(graph, node2idx):
    node_size = graph.number_of_nodes()
    A_data = []
    A_row_index = []
    A_col_index = []

    for edge in graph.edges():
        v1, v2 = edge
        edge_weight = graph[v1][v2].get('weight', 1)

        A_data.append(edge_weight)
        A_row_index.append(node2idx[v1])
        A_col_index.append(node2idx[v2])

    A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
    A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                       shape=(node_size, node_size))

    D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
    L = D - A_
    return A, L


class SDNE(object):
    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4):

        self.graph = graph
        # self.g.remove_edges_from(self.g.selfloop_edges())
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)

        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        self.A, self.L = create_A_L(self.graph, self.node2idx)  # Adj Matrix,L Matrix
        self.reset_model()
        self.inputs = [self.A, self.L]
        self._embeddings = {}

    def reset_model(self, opt='adam'):
        self.model, self.emb_model = create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1, l2=self.nu2)
        self.model.compile(opt, [l_2nd(self.beta), l_1st(self.alpha)])
        self.get_embeddings()

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}), set batch_size = {1}'.format(batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()],
                                  batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose,
                                  shuffle=False, )
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(i * batch_size, min((i + 1) * batch_size, self.node_size))
                    A_train = self.A[index, :].todense()
                    L_mat_train = self.L[index][:, index].todense()
                    inp = [A_train, L_mat_train]
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses = losses / steps_per_epoch

                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                    print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}'.format(
                        epoch_time, losses[0], losses[1], losses[2]))
            return hist

    def evaluate(self, ):
        return self.model.evaluate(x=self.inputs, y=self.inputs, batch_size=self.node_size)

    def get_embeddings(self):
        self._embeddings = {}
        embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding

        return self._embeddings




import torch
import torch.nn as nn
import torch.utils.data as data
import os


class SDNEModel(nn.Module):
    def __init__(self, node_size, hidden_sizes):
        super(SDNEModel, self).__init__()

        hidden_sizes.insert(0, node_size)

        encoder = []
        for i in range(1, len(hidden_sizes)):
            encoder += [nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                        nn.ReLU(inplace=True)]

        decoder = []
        for i in range(len(hidden_sizes)-1, 0, -1):
            decoder += [nn.Linear(hidden_sizes[i], hidden_sizes[i-1]),
                        nn.ReLU(inplace=True)]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, A, L=None):
        Y = self.encoder(A)
        if L is None:
            return Y
        A_ = self.decoder(Y)
        return A_, Y


class L_2nd(nn.Module):
    def __init__(self, beta):
        super(L_2nd, self).__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        b = torch.ones_like(y_true)
        b[y_true != 0] = self.beta
        x = ((y_true-y_pred)*b)**2
        t = torch.sum(x, dim=-1)
        return torch.mean(t)


class L_1st(nn.Module):
    def __init__(self, alpha):
        super(L_1st, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        Y = y_pred
        L = y_true
        batch_size = Y.shape[0]
        return 2*self.alpha*torch.trace(torch.mm(torch.mm(Y.transpose(0, 1), L), Y)) / batch_size


# class L_overall(nn. Module):
#     def __init__(self, alpha, beta):
#         super(L_overall, self).__init__()
#         self.L_1st = L_1st(alpha)
#         self.L_2nd = L_2nd(beta)
#
#     def forward(self, y_pred, y_true):
#         return self.L_1st(y_pred, y_true) + self.L_2nd(y_pred, y_true)


class L1_penalty(nn.Module):
    def __init__(self, model, nu):
        super(L1_penalty, self).__init__()
        self.model = model
        self.nu = nu

    def forward(self):
        return self.nu * sum([torch.norm(param, 1) for param in self.model.parameters()])


class SDNEDataset(data.Dataset):
    def __init__(self, node_size):
        self.indexes = list(range(node_size))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.indexes[idx]


class SDNE2:
    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5, nu1=1e-5, nu2=1e-4, epochs=1, batch_size=1024, gpu=''):
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.node_size = self.graph.number_of_nodes()
        self._embeddings = {}

        model = SDNEModel(self.node_size, hidden_size)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = max([4*torch.cuda.device_count(), 4])
        if self.device == 'cuda':
            device_count = torch.cuda.device_count()
            torch.backends.cudnn.benchmark = True
            print(f"Let's use {device_count} GPUs!")
        model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), weight_decay=nu2)
        self.criterion1 = L_1st(alpha)
        self.criterion2 = L_2nd(beta)
        self.L1_regula = L1_penalty(model, nu1)

        self.A, self.L = create_A_L(self.graph, self.node2idx)
        self.dataLoader = data.DataLoader(SDNEDataset(self.node_size), batch_size=batch_size,
                                          shuffle=False, num_workers=self.num_workers)
        self.epochs = epochs

    def train(self):
        for i in range(self.epochs):
            print(f'Epoch: {i+1}/{self.epochs}')
            epoch_loss, epoch_loss_1st, epoch_loss_2nd = self.train_epoch()
            print(f'Epoch loss: {epoch_loss:.3f}, 1st loss: {epoch_loss_1st:.3f}, 2nd loss: {epoch_loss_2nd:.3f}')

    def train_epoch(self):
        epoch_loss = 0
        epoch_loss_1st = 0
        epoch_loss_2nd = 0
        self.model.train()
        for _, index in enumerate(self.dataLoader):
            A = torch.from_numpy(self.A[index].toarray()).to(torch.float)
            L = torch.from_numpy(self.L[index][:, index].toarray()).to(torch.float)
            A = A.to(self.device)
            L = L.to(self.device)
            A_, Y = self.model(A, L)
            loss_1st = self.criterion1(Y, L)
            loss_2nd = self.criterion2(A_, A)
            l1_regula = self.L1_regula()
            loss = loss_1st + loss_2nd + l1_regula
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss_1st += loss_1st.item()
            epoch_loss_2nd += loss_2nd.item()
            epoch_loss += loss.item()
        epoch_loss /= len(self.dataLoader)
        epoch_loss_1st /= len(self.dataLoader)
        epoch_loss_2nd /= len(self.dataLoader)

        return epoch_loss, epoch_loss_1st, epoch_loss_2nd

    def get_embeddings(self):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for _, index in enumerate(self.dataLoader):
                A = torch.from_numpy(self.A[index].toarray()).to(torch.float)
                L = torch.from_numpy(self.L[index][:, index].toarray()).to(torch.float)
                A = A.to(self.device)
                L = L.to(self.device)
                Y = self.model(A)
                embeddings.append(Y)
        embeddings = torch.cat(embeddings, dim=0).detach().to('cpu').numpy()

        look_back = self.idx2node
        for idx, embedding in enumerate(embeddings):
            node = look_back[idx]
            self._embeddings[node] = embedding

        return self._embeddings