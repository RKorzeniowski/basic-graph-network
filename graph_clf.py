import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt

# from sklearn.manifold import TSNE
# def visualize(h, color):
#     z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
#
#     plt.figure(figsize=(10,10))
#     plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#     plt.show()


from torch_geometric.utils import to_networkx



def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # GraphConv
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 2)
        self.lin2 = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        h = self.lin1(x)
        x = self.lin2(h)
        return x, h


def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
         out, h = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.


def test(loader, plot):
     model.eval()

     hidden, color = [], []
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out, h = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         hidden.append(h)
         color.append(data.y)

     if plot:
         hidden = torch.cat(hidden, dim=0)
         color = torch.cat(color, dim=0)
         visualize(hidden, color)
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


dataset = TUDataset(root='data/TUDataset', name='MUTAG')

data = dataset[0]
G = to_networkx(data, to_undirected=True)
nx.draw(G, node_size=10, with_labels=True)

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 10):
    train()
    train_acc = test(train_loader, plot=False)
    test_acc = test(test_loader, plot=True)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
