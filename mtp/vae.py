### Originally from https://www.vadimborisov.com/conditional-variational-autoencoder-cvae.html
from collections import defaultdict

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# cuda setup, if do not GPU
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}

# hyper params
batch_size = 64
latent_size = 20
epochs = 20


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False, **kwargs)


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fcl = nn.Linear(latent_size, class_size)

        self.fc3 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        # inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(x))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        # inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        c = self.sigmoid(self.fcl(z))
        h3 = self.elu(self.fc3(z))
        return self.sigmoid(self.fc4(h3)), c

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        d, c = self.decode(z)
        return d, c, mu, logvar

# create a CVAE model
model = CVAE(28*28, latent_size, 10).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, recon_c, c, mu, logvar):
    BCE1 = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE2 = F.binary_cross_entropy(recon_c, c, reduction='sum')
    beta = 1.
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE1 + BCE2 + beta * KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        labels = one_hot(labels, 10)
        recon_batch, recon_label, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, recon_label, labels, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels_) in enumerate(test_loader):
            data, labels_ = data.to(device), labels_.to(device)
            labels = one_hot(labels_, 10)
            recon_batch, recon_label, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, recon_label, labels, mu, logvar).detach().cpu().numpy()
            if i == 0:
                pred_labels = []
                for j in range(10):
                    _, c = model.decode(model.reparameterize(mu, logvar))
                    pred_labels.append(torch.argmax(c, dim=1))
                pred_labels = torch.stack(pred_labels, dim=-1)
                for tar, row in zip(labels_, pred_labels):
                    counts = defaultdict(int)
                    for v in row:
                        counts[v.item()] += 1
                    print(tar.item(), [counts[i] for i in range(10)])

                # for l1, l2, l3 in zip(labels_, torch.argmax(recon_label, dim=1), torch.argmax(c, dim=1)):
                #     print(l1.item(), l2.item(), l3.item())
                n = min(data.size(0), 5)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(epoch) + '.png', nrow=n)
                num_sample = labels_.shape[0]
                correct = sum([int(l1.item() == l2.item()) for l1, l2 in zip(labels_, torch.argmax(recon_label, dim=1))])
                print('train', correct / num_sample)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            c = torch.eye(10, 10).cuda()
            sample = torch.randn(10, 20).to(device)
            x, c = model.decode(sample)
            sample = x.cpu()
            print('test', sum([int(i1 == i2.item()) for i1, i2 in zip(range(10), torch.argmax(c, dim=1))]) / 10)
            save_image(sample.view(10, 1, 28, 28),
                       'sample_' + str(epoch) + '.png')
