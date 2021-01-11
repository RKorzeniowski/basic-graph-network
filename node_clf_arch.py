import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GraphConv, GATConv, RENet
from torch_geometric.datasets import Planetoid

arch_mapping = {
    "conv": GraphConv,
    "attention": GATConv,
    "rnn": RENet,
}

# CNN
# GraphConv https://arxiv.org/abs/1810.02244
# GCNConv https://arxiv.org/abs/1609.02907

# Attention
# GATConv https://arxiv.org/abs/1710.10903
# AGNNConv https://arxiv.org/abs/1803.03735


EPOCHS = 100
PLOT = True


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = arch_mapping["conv"](dataset.num_node_features, 16)
        self.conv2 = arch_mapping["conv"](16, dataset.num_classes)

        # self.conv1 = arch_mapping["attention"](dataset.num_node_features, 16)
        # self.conv2 = arch_mapping["attention"](16, dataset.num_classes)

        # self.conv1 = arch_mapping["rnn"](dataset.num_node_features, 16)
        # self.conv2 = arch_mapping["rnn"](16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
        'node_size': 30,
        'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


def train(data, epochs, plot=False):
    train_accuracies, test_accuracies = list(), list()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc = test(data)
        test_acc = test(data, train=False)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    print('Best test accuracy:', max(test_accuracies))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    plot_dataset(dataset)

    device = torch.device('cpu')
    model = Net(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train(data, epochs=EPOCHS, plot=PLOT)
