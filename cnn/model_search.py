import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

device = torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(self, C, stride, learnable_bn):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, learnable_bn)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, gumbel=False):
        if gumbel:
            clist = []
            for j, weight in enumerate(weights):
                if abs(weight) > 1e-10:
                    clist.append(weights[j] * self._ops[j](x))
            if len(clist) == 1:
                return clist[0]
            else:
                return sum(clist)
        else:
            return sum(w * op(x) for w, op in zip(weights, self._ops))

    # def forward_gdas(self, x, weights, index):
    #     return self._ops[index](x) * weights[index]


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, learnable_bn):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
          for j in range(2+i):
            stride = 2 if reduction and j < 2 else 1
            op = MixedOp(C, stride, learnable_bn)
            self._ops.append(op)

        # self.edges = nn.ModuleDict()
        # for i in range(self._steps):
        #     for j in range(2 + i):
        #         node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
        #         stride = 2 if reduction and j < 2 else 1
        #         op = MixedOp(C, stride, learnable_bn)
        #         self.edges[node_str] = op
        # self.edge_keys = sorted(list(self.edges.keys()))
        # self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        # self.num_edges = len(self.edges)

    def forward(self, s0, s1, weights, gumbel=False):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j], gumbel) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

    # def forward_gdas(self, s0, s1, weights):
    #     s0 = self.preprocess0(s0)
    #     s1 = self.preprocess1(s1)
    #
    #     states = [s0, s1]
    #     for i in range(self._steps):
    #         clist = []
    #         for j, h in enumerate(states):
    #             node_str = '{:}<-{:}'.format(i, j)
    #             op = self.edges[node_str]
    #             weights = weightss[self.edge2index[node_str]]
    #             index = indexs[self.edge2index[node_str]].item()
    #             clist.append(op.forward_gdas(h, weights, index))
    #         states.append(sum(clist))
    #
    #     return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, learnable_bn=False, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._learnable_bn = learnable_bn

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, learnable_bn)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self.tau = 5

    def new(self):
        # model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._learnable_bn).to(device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, gumbel=False):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if gumbel:
                    weights = F.gumbel_softmax(self.alphas_reduce, self.tau, True)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if gumbel:
                    weights = F.gumbel_softmax(self.alphas_normal, self.tau, True)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell.forward(s0, s1, weights, gumbel)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    # def forward_gdas(self, input):
        # s0 = s1 = self.stem(input)
        # for i, cell in enumerate(self.cells):
        #     if cell.reduction:
        #         weights = F.gumbel_softmax(self.alphas_reduce, self.tau, True)
        #     else:
        #         weights = F.gumbel_softmax(self.alphas_normal, self.tau, True)
        #     s0, s1 = s1, cell.forward(s0, s1, weights)
        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits

        # def get_gumbel_prob(xins):
        #     while True:
        #         gumbels = -torch.empty_like(xins).exponential_().log()
        #         logits = (xins.log_softmax(dim=1) + gumbels) / self.tau
        #         probs = nn.functional.softmax(logits, dim=1)
        #         index = probs.max(-1, keepdim=True)[1]
        #         one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        #         hardwts = one_h - probs.detach() + probs
        #         if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
        #             continue
        #         else:
        #             break
        #     return hardwts, index
        #
        # normal_hardwts, normal_index = get_gumbel_prob(self.alphas_normal)
        # reduce_hardwts, reduce_index = get_gumbel_prob(self.alphas_reduce)
        #
        # s0 = s1 = self.stem(input)
        # for i, cell in enumerate(self.cells):
        #     if cell.reduction:
        #         hardwts, index = reduce_hardwts, reduce_index
        #     else:
        #         hardwts, index = normal_hardwts, normal_index
        #     s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
        # # out = self.lastact(s1)
        # # out = self.global_pooling(out)
        # out = self.global_pooling(s1)
        # out = out.view(out.size(0), -1)
        # logits = self.classifier(out)
        #
        # return out, logits

    def _loss(self, input, target, gumbel=False):
        logits = self.forward(input, gumbel)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        # self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        # self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).to(device), requires_grad=True)
        self.alphas_normal = [Variable(1e-3 * torch.randn(num_ops).to(device), requires_grad=True) for edge in range(k)]
        # self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).to(device), requires_grad=True)
        self.alphas_reduce = [Variable(1e-3 * torch.randn(num_ops).to(device), requires_grad=True) for edge in range(k)]
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
