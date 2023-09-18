# -*- coding: UTF-8 -*-
# Date: 2022/09/19

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args, vocab=258, kernel_size=[2, 3, 5, 7]):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(258, args.nhid)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.num_filters, (k, args.nhid)) for k in kernel_size]
        )
        # self.dropout = nn.Dropout(args.dropout)
        self.norm = nn.BatchNorm1d(args.num_filters * len(kernel_size))
        self.fc = nn.Sequential(
            nn.Linear(
                args.num_filters * len(kernel_size), args.nhid * len(kernel_size) // 2
            ),
            nn.BatchNorm1d(args.nhid * len(kernel_size) // 2),
            nn.ELU(),
            nn.Linear(args.nhid * len(kernel_size) // 2, args.nhid),
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        out = self.norm(out)
        out = self.fc(out)
        return out


class DPCNN(nn.Module):
    def __init__(self, args, vocab=258):
        super().__init__()
        self.embedding = nn.Embedding(vocab, args.nhid)
        self.channel_size = args.num_filters
        self.conv_region = nn.Conv2d(1, self.channel_size, (3, args.nhid), stride=1)
        self.conv = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.channel_size, args.nhid)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] >= 2:
            x = self._block(x)

        x = x.view(batchsize, -1)  # [batch_size, num_filters(250)]

        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


class LSTMModel(nn.Module):
    def __init__(self, args, d_feat=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 64)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :])
