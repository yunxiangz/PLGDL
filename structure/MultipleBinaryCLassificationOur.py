import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from torchdrug.layers import functional
class MultipleBinaryClassificationOur(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0):
        super(MultipleBinaryClassificationOur, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [len(task)])

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)

        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return output

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
