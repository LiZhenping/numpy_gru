# -*- coding: utf-8 -*-
"""
rnn_text_generator_word_level.py

该模块实现了一个基于 NumPy 的词级别文本生成器，使用 RNN 模型。
代码包含训练和文本生成功能，适用于教学和理解 RNN 的工作原理。

Created on Mon Nov 25 11:17:34 2019
@author: lizhenping
"""

import numpy as np
import re
from collections import Counter

class Tokenizer:
    """
    词级别的分词器，负责将文本转换为词语索引序列，以及索引序列转换为文本。
    """
    def __init__(self, text, max_vocab_size=None):
        self.special_tokens = ['<PAD>', '<UNK>', '<EOS>']  # 特殊标记
        self.max_vocab_size = max_vocab_size
        self.build_vocab(text)

    def build_vocab(self, text):
        # 使用正则表达式分词，保留标点符号
        words = re.findall(r'\w+|[^\s\w]+', text)
        word_counts = Counter(words)

        if self.max_vocab_size:
            # 根据词频排序，取前 max_vocab_size 个词
            most_common = word_counts.most_common(self.max_vocab_size - len(self.special_tokens))
            vocab = self.special_tokens + [word for word, _ in most_common]
        else:
            vocab = self.special_tokens + list(word_counts.keys())

        self.word_to_ix = {word: i for i, word in enumerate(vocab)}
        self.ix_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def text_to_sequence(self, text):
        # 将文本转换为词语索引序列
        words = re.findall(r'\w+|[^\s\w]+', text)
        sequence = []
        for word in words:
            sequence.append(self.word_to_ix.get(word, self.word_to_ix['<UNK>']))
        sequence.append(self.word_to_ix['<EOS>'])
        return sequence

    def sequence_to_text(self, sequence):
        # 将词语索引序列转换为文本
        words = [self.ix_to_word.get(idx, '<UNK>') for idx in sequence]
        text = ''.join(words)  # 对于中文不需要空格分隔
        return text

class Module:
    """
    模块基类，所有神经网络模块的父类。
    提供参数管理、梯度清零、参数初始化功能。
    """
    def __init__(self):
        self.parameters = []
        self.gradients = []

    def zero_grad(self):
        """
        将所有参数的梯度清零。
        """
        for grad in self.gradients:
            grad.fill(0)

    def init_weights(self, init_range):
        """
        初始化模型权重，服从[-init_range, init_range]的均匀分布。
        """
        for param in self.parameters:
            param[:] = np.random.uniform(-init_range, init_range, size=param.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    # y = sigmoid(x)
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    # y = tanh(x)
    return 1 - y ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class RNNCell(Module):
    """
    实现单个 RNN 单元，包括前向和反向传播。
    """
    def __init__(self, input_size, hidden_size, init_range=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化权重和偏置
        self.W_ih = np.random.uniform(-init_range, init_range, (hidden_size, input_size))
        self.W_hh = np.random.uniform(-init_range, init_range, (hidden_size, hidden_size))
        self.b_h = np.zeros((hidden_size, 1))

        # 存储参数和对应的梯度
        self.parameters = [self.W_ih, self.W_hh, self.b_h]
        self.gradients = [np.zeros_like(param) for param in self.parameters]

    def forward(self, x, h_prev):
        """
        前向传播

        参数:
        - x: 当前时间步的输入，形状为 (input_size, batch_size)
        - h_prev: 前一时间步的隐藏状态，形状为 (hidden_size, batch_size)

        返回:
        - h_next: 当前时间步的隐藏状态
        """
        self.x = x
        self.h_prev = h_prev

        # 计算隐藏状态
        self.h_next = tanh(np.dot(self.W_ih, x) + np.dot(self.W_hh, h_prev) + self.b_h)
        return self.h_next

    def backward(self, dh_next, grad_clip=5):
        """
        反向传播

        参数:
        - dh_next: 当前时间步的隐藏状态梯度，形状为 (hidden_size, batch_size)
        - grad_clip: 梯度裁剪阈值，默认为 5

        返回:
        - dx: 对输入 x 的梯度
        - dh_prev: 对前一隐藏状态 h_prev 的梯度
        """
        # 计算 tanh 的梯度
        dtanh_h = dh_next * dtanh(self.h_next)

        # 计算参数梯度
        dW_ih = np.dot(dtanh_h, self.x.T)
        dW_hh = np.dot(dtanh_h, self.h_prev.T)
        db_h = np.sum(dtanh_h, axis=1, keepdims=True)

        # 裁剪梯度，避免梯度爆炸
        dW_ih = np.clip(dW_ih, -grad_clip, grad_clip)
        dW_hh = np.clip(dW_hh, -grad_clip, grad_clip)
        db_h = np.clip(db_h, -grad_clip, grad_clip)

        # 计算对输入和前一隐藏状态的梯度
        dx = np.dot(self.W_ih.T, dtanh_h)
        dh_prev = np.dot(self.W_hh.T, dtanh_h)

        # 存储梯度
        self.gradients[0] += dW_ih
        self.gradients[1] += dW_hh
        self.gradients[2] += db_h

        return dx, dh_prev

class RNNLayer(Module):
    """
    RNN 层，由多个 RNN 单元组成。
    """
    def __init__(self, input_size, hidden_size, num_layers=1, init_range=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size if i == 0 else hidden_size, hidden_size, init_range)
                          for i in range(num_layers)]

        # 存储参数和梯度
        self.parameters = []
        self.gradients = []
        for rnn_cell in self.rnn_cells:
            self.parameters.extend(rnn_cell.parameters)
            self.gradients.extend(rnn_cell.gradients)

    def forward(self, x, h0):
        """
        前向传播

        参数:
        - x: 输入序列，形状为 (input_size, seq_len, batch_size)
        - h0: 初始隐藏状态列表，长度为 num_layers，每个元素形状为 (hidden_size, batch_size)

        返回:
        - outputs: 每个时间步的输出，形状为 (hidden_size, seq_len, batch_size)
        - h_n: 最后一个时间步的隐藏状态列表
        """
        seq_len = x.shape[1]
        batch_size = x.shape[2]
        h = [h0_layer.copy() for h0_layer in h0]
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]  # 当前时间步的输入，形状 (input_size, batch_size)
            for i, rnn_cell in enumerate(self.rnn_cells):
                h_prev = h[i]
                h_next = rnn_cell.forward(xt, h_prev)
                h[i] = h_next
                xt = h_next  # 当前层的输出作为下一层的输入
            outputs.append(h[-1].reshape(self.hidden_size, 1, batch_size))  # 收集最后一层的输出

        outputs = np.concatenate(outputs, axis=1)  # (hidden_size, seq_len, batch_size)
        return outputs, h

    def backward(self, doutputs, dh_n, grad_clip=5):
        """
        反向传播

        参数:
        - doutputs: 每个时间步的输出梯度，形状为 (hidden_size, seq_len, batch_size)
        - dh_n: 最后一个时间步的隐藏状态梯度列表
        - grad_clip: 梯度裁剪阈值，默认为 5

        返回:
        - dx: 输入序列的梯度，形状为 (input_size, seq_len, batch_size)
        """
        seq_len = doutputs.shape[1]
        batch_size = doutputs.shape[2]
        dh_next = [dh.copy() for dh in dh_n]
        dx = []

        for t in reversed(range(seq_len)):
            dh = doutputs[:, t, :]  # (hidden_size, batch_size)
            for i in reversed(range(self.num_layers)):
                rnn_cell = self.rnn_cells[i]
                dh = dh + dh_next[i]
                dx_step, dh_prev = rnn_cell.backward(dh, grad_clip)
                dh_next[i] = dh_prev
                dh = dx_step  # 当前层的 dx_step 作为上一层的 dh
            dx.insert(0, dx_step.reshape(self.input_size, 1, batch_size))

        dx = np.concatenate(dx, axis=1)  # 拼接各时间步的输入梯度，形状 (input_size, seq_len, batch_size)
        return dx  # 保持原形状返回

class Embedding(Module):
    """
    嵌入层，将词语索引转换为词向量。
    """
    def __init__(self, vocab_size, embedding_dim, init_range=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 初始化嵌入矩阵
        self.embedding_matrix = np.random.uniform(-init_range, init_range, (vocab_size, embedding_dim))

        # 存储参数和梯度
        self.parameters = [self.embedding_matrix]
        self.gradients = [np.zeros_like(self.embedding_matrix)]

    def forward(self, inputs):
        """
        前向传播

        参数:
        - inputs: 词语索引序列，形状为 (seq_len, batch_size)

        返回:
        - outputs: 词向量序列，形状为 (embedding_dim, seq_len, batch_size)
        """
        self.inputs = inputs
        outputs = self.embedding_matrix[inputs]  # shape: (seq_len, batch_size, embedding_dim)
        outputs = outputs.transpose(2, 0, 1)  # 转换为 (embedding_dim, seq_len, batch_size)
        return outputs

    def backward(self, doutputs):
        """
        反向传播

        参数:
        - doutputs: 词向量序列的梯度，形状为 (embedding_dim, seq_len, batch_size)

        返回:
        - 无需返回输入梯度
        """
        doutputs = doutputs.transpose(1, 2, 0)  # 转换为 (seq_len, batch_size, embedding_dim)
        for i in range(self.inputs.shape[0]):  # seq_len
            for j in range(self.inputs.shape[1]):  # batch_size
                idx = self.inputs[i, j]
                self.gradients[0][idx] += doutputs[i, j]
        # 返回的梯度不传递给前面的层
        return None

class Linear(Module):
    """
    线性层，实现 y = Wx + b。
    """
    def __init__(self, input_size, output_size, bias=True, init_range=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        # 初始化权重和偏置
        self.W = np.random.uniform(-init_range, init_range, (output_size, input_size))
        self.b = np.zeros((output_size, 1)) if self.use_bias else None

        # 存储参数和梯度
        self.parameters = [self.W] if not self.use_bias else [self.W, self.b]
        self.gradients = [np.zeros_like(param) for param in self.parameters]

    def forward(self, inputs):
        """
        前向传播

        参数:
        - inputs: 输入，形状为 (input_size, batch_size)

        返回:
        - outputs: 输出，形状为 (output_size, batch_size)
        """
        self.inputs = inputs
        outputs = np.dot(self.W, inputs)
        if self.use_bias:
            outputs += self.b
        return outputs

    def backward(self, doutputs):
        """
        反向传播

        参数:
        - doutputs: 输出的梯度，形状为 (output_size, batch_size)

        返回:
        - dinputs: 输入的梯度，形状为 (input_size, batch_size)
        """
        dinputs = np.dot(self.W.T, doutputs)
        dW = np.dot(doutputs, self.inputs.T)
        self.gradients[0] += dW
        if self.use_bias:
            db = np.sum(doutputs, axis=1, keepdims=True)
            self.gradients[1] += db
        return dinputs

class CrossEntropyLoss:
    """
    交叉熵损失函数
    """
    def __init__(self):
        pass

    def forward(self, inputs, targets, reduction='mean'):
        """
        前向计算损失值

        参数:
        - inputs: 模型的输出，形状为 (num_classes, batch_size)
        - targets: 目标类别索引，形状为 (batch_size,)
        - reduction: 损失归约方式，可选 'mean' 或 'sum'，默认为 'mean'

        返回:
        - loss: 标量，平均或求和后的损失值
        """
        self.inputs = inputs
        self.targets = targets
        num_classes, batch_size = inputs.shape

        # 计算每个样本的对数 softmax
        shifted_logits = inputs - np.max(inputs, axis=0, keepdims=True)
        log_probs = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis=0, keepdims=True))
        self.probs = np.exp(log_probs)
        loss = -log_probs[self.targets, range(batch_size)]
        if reduction == 'mean':
            return np.mean(loss)
        elif reduction == 'sum':
            return np.sum(loss)

    def backward(self):
        """
        反向传播，计算输入的梯度

        返回:
        - dinputs: 模型输出的梯度，形状与 inputs 相同
        """
        batch_size = self.inputs.shape[1]
        dinputs = self.probs.copy()
        dinputs[self.targets, range(batch_size)] -= 1
        dinputs /= batch_size
        return dinputs

class AdamOptimizer:
    """
    Adam 优化器
    """
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(param) for param in parameters]  # 一阶矩
        self.v = [np.zeros_like(param) for param in parameters]  # 二阶矩
        self.t = 0  # 时间步

    def step(self, gradients):
        """
        应用梯度，更新参数，并更新时间步长。

        参数:
        - gradients: 与 parameters 对应的梯度列表
        """
        self.t += 1  # 在参数更新之前递增时间步长

        for idx, (param, grad) in enumerate(zip(self.parameters, gradients)):
            # 应用权重衰减
            if self.weight_decay != 0:
                grad += self.weight_decay * param

            # 更新一阶矩和二阶矩估计
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)

            # 修正一阶矩和二阶矩的偏差
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            # 更新参数
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class LanguageModel(Module):
    """
    语言模型，包括词嵌入层、RNN 层和线性输出层。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, init_range=0.1, bias=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, embedding_dim, init_range)
        self.rnn = RNNLayer(embedding_dim, hidden_size, num_layers, init_range)
        self.output_layer = Linear(hidden_size, vocab_size, bias, init_range)

        # 存储参数和梯度
        self.parameters = self.embedding.parameters + self.rnn.parameters + self.output_layer.parameters
        self.gradients = self.embedding.gradients + self.rnn.gradients + self.output_layer.gradients

    def forward(self, inputs, h0):
        """
        前向传播

        参数:
        - inputs: 词语索引序列，形状为 (seq_len, batch_size)
        - h0: 初始隐藏状态列表，长度为 num_layers，每个元素形状为 (hidden_size, batch_size)

        返回:
        - outputs: 每个时间步的输出，形状为 (vocab_size, seq_len * batch_size)
        - h_n: 最后一个时间步的隐藏状态列表
        """
        self.seq_len, self.batch_size = inputs.shape
        embeddings = self.embedding.forward(inputs)
        outputs, h_n = self.rnn.forward(embeddings, h0)
        outputs = outputs.reshape(self.hidden_size, -1)  # (hidden_size, seq_len * batch_size)
        outputs = self.output_layer.forward(outputs)
        return outputs, h_n

    def backward(self, doutputs, dh_n):
        """
        反向传播

        参数:
        - doutputs: 每个时间步的输出梯度，形状为 (vocab_size, seq_len * batch_size)
        - dh_n: 最后时间步的隐藏状态梯度列表

        返回:
        - 无需返回输入梯度
        """
        d_rnn_outputs = self.output_layer.backward(doutputs)
        d_rnn_outputs = d_rnn_outputs.reshape(self.hidden_size, self.seq_len, self.batch_size)
        d_embeddings = self.rnn.backward(d_rnn_outputs, dh_n)
        if d_embeddings is not None:
            self.embedding.backward(d_embeddings)

class LanguageModelTrainer:
    """
    语言模型训练器
    """
    def __init__(self, model, loss_func, optimizer):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

    def train(self, inputs, targets, h0, grad_clip=5):
        """
        训练一个批次的数据

        参数:
        - inputs: 输入词语索引序列，形状为 (seq_len, batch_size)
        - targets: 目标词语索引序列，形状为 (seq_len, batch_size)
        - h0: 初始隐藏状态列表
        - grad_clip: 梯度裁剪阈值，默认为 5

        返回:
        - loss: 标量，平均损失值
        """
        self.model.zero_grad()
        outputs, h_n = self.model.forward(inputs, h0)
        loss = self.loss_func.forward(outputs, targets.flatten())
        dloss = self.loss_func.backward()
        dh_n = [np.zeros_like(h) for h in h_n]
        self.model.backward(dloss, dh_n)

        # 裁剪梯度
        clipped_gradients = [np.clip(grad, -grad_clip, grad_clip) for grad in self.model.gradients]

        # 应用梯度
        self.optimizer.step(clipped_gradients)
        return loss

def generate_dummy_data():
    # 为了更好地模拟实际文本，包含空格分隔的中文词语和标点符号
    text = '你好世界。 机器学习是有趣的。 你好机器。 '
    return text

def train_model():
    # 生成数据
    text = generate_dummy_data()
    tokenizer = Tokenizer(text)
    data = tokenizer.text_to_sequence(text)

    # 初始化模型和优化器
    vocab_size = tokenizer.vocab_size
    embedding_dim = 10
    hidden_size = 20
    num_layers = 1
    model = LanguageModel(vocab_size, embedding_dim, hidden_size, num_layers)
    loss_func = CrossEntropyLoss()
    optimizer = AdamOptimizer(model.parameters)
    trainer = LanguageModelTrainer(model, loss_func, optimizer)

    # 准备数据
    seq_len = len(data) - 1
    batch_size = 1
    inputs = np.array([data[:-1]]).T  # (seq_len, batch_size)
    targets = np.array([data[1:]]).T  # (seq_len, batch_size)
    h0 = [np.zeros((hidden_size, batch_size)) for _ in range(num_layers)]

    # 训练
    epochs = 100000
    for epoch in range(epochs):
        loss = trainer.train(inputs, targets, h0)
        if (epoch + 1) % 100 == 0:
            print(f'第 {epoch+1} 轮训练，损失值: {loss:.4f}')

    return model, tokenizer

def generate_text(model, tokenizer, seed_text, max_len=20, temperature=1.0):
    generated_text = seed_text
    h = [np.zeros((model.hidden_size, 1)) for _ in range(model.num_layers)]

    # 使用种子文本初始化隐藏状态
    seed_sequence = tokenizer.text_to_sequence(seed_text)
    for word_id in seed_sequence[:-1]:
        inputs = np.array([[word_id]])  # (seq_len=1, batch_size=1)
        _, h = model.forward(inputs, h)

    last_word_id = seed_sequence[-1]

    for _ in range(max_len):
        inputs = np.array([[last_word_id]])  # (seq_len=1, batch_size=1)
        outputs, h = model.forward(inputs, h)
        logits = outputs[:, -1]  # (vocab_size,)
        logits = logits / temperature
        probs = np.exp(logits - np.max(logits))  # 防止溢出
        probs /= np.sum(probs)
        next_word_id = np.random.choice(len(probs), p=probs)
        next_word = tokenizer.ix_to_word[next_word_id]
        if next_word == '<EOS>':
            break
        generated_text += next_word
        last_word_id = next_word_id
    return generated_text

# 主程序
if __name__ == "__main__":
    # 训练模型
    model, tokenizer = train_model()

    # 示例生成
    seed_text = '你好'
    generated = generate_text(model, tokenizer, seed_text)
    print(f'生成的文本: {generated}')
