import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GCN, GIN
from .text import DPCNN, TextCNN, LSTMModel
from .wide import Wide
from .image import resnet18


class BinaryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.byte_nn = TextCNN(args, kernel_size=[2, 3, 5, 8])
        # self.asm_nn = TextCNN(args, vocab=129, kernel_size=[3, 5, 9, 15])
        self.byte_nn = DPCNN(args)
        self.asm_nn = DPCNN(args, vocab=129)
        self.integer_nn = TextCNN(args, kernel_size=[1, 2, 3, 4])
        self.graph_embedding = GIN(args)

        channel = 0
        if self.args.bytecode:
            channel += 1
        if self.args.asm:
            channel += 1
        if self.args.graph:
            channel += 1
        if self.args.integer:
            channel += 1
        if self.args.wide:
            self.wide = Wide()
            channel += 1
        if self.args.image:
            self.image = resnet18()
            channel += 1

        self.liner = nn.Sequential(
            nn.Linear(args.nhid * channel, args.nhid * channel // 2),
            nn.BatchNorm1d(args.nhid * channel // 2),
            nn.ELU(),
            nn.Linear(args.nhid * channel // 2, args.nhid),
            nn.BatchNorm1d(args.nhid),
            nn.ELU(),
            nn.Linear(args.nhid, args.nhid),
        )

    def forward(self, x):
        outs = []
        if self.args.bytecode:
            code_out = self.byte_nn(x["code"])
            outs.append(code_out)
        if self.args.asm:
            asm_out = self.asm_nn(x["asm"])
            outs.append(asm_out)
        if self.args.graph:
            graph_out = self.graph_embedding(x["graph"])
            outs.append(graph_out)
        if self.args.wide:
            wide_out = self.wide(x["wide"])
            outs.append(wide_out)
        if self.args.integer:
            integer_out = self.integer_nn(x["integer"])
            outs.append(integer_out)
        if self.args.image:
            image_out = self.image(x["image"])
            outs.append(image_out)

        out = torch.cat(outs, dim=1)
        out = self.liner(out)

        return out
