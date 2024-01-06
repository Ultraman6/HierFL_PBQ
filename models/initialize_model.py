# Interface between models and the clients
# Include intialization, training for one iteration and test function
from models.celeba_gan import GAN, Generator, Discriminator
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv, cifar_cnn_3conv_specific, cifar_cnn_3conv_shared
from models.cifar_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet18_YWX
from models.mnist_cnn import mnist_lenet, mnist_cnn
from models.mnist_logistic import LogisticRegression_MNIST
import torch.optim as optim
import torch.nn as nn

# from models.shakespare_lstm import LSTMModel
from models.shakespare_rnn import RNNModel
from models.synthetic_logistic import LogisticRegression_SYNTHETIC
# following import is used for tesing the function of this part, they can be deleted if you delete the main() funciton
from options import args_parser
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import dirname, abspath, join
from torch.autograd import Variable
from tqdm import tqdm


class MTL_Model(object):
    def __init__(self, shared_layers, specific_layers, learning_rate, lr_decay, lr_decay_epoch, momentum, weight_decay, criterion):
        self.shared_layers = shared_layers
        self.specific_layers = specific_layers
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.momentum = momentum
        self.weight_decay = weight_decay
        #   construct the parameter
        param_dict = [{"params": self.shared_layers.parameters()}]
        if self.specific_layers:
            param_dict += [{"params": self.specific_layers.parameters()}]
        self.optimizer = optim.SGD(params=param_dict,
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.criterion = criterion

    def exp_lr_sheduler(self, epoch):
        """"""

        if (epoch + 1) % self.lr_decay_epoch:
            return None
        for param_group in self.optimizer.param_groups:
            # print(f'epoch{epoch}')
            param_group['lr'] *= self.lr_decay
            return None

    def step_lr_scheduler(self, epoch):
        if epoch < 150:
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.1
        elif epoch >= 150 and epoch < 250:
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.01
        elif epoch >= 250:
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.001

    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])

    def optimize_model(self, input_batch, label_batch, device):
        if isinstance(self.shared_layers, GAN):
            # GAN Training Logic
            # Assuming self.shared_layers has separate generator and discriminator
            real_images = input_batch
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            # Train Discriminator
            self.optimizer.zero_grad()
            # print(real_labels.shape)
            # print(self.shared_layers.discriminator(real_images).shape)
            real_loss = self.criterion(self.shared_layers.discriminator(real_images), real_labels)
            noise = torch.randn(real_images.size(0), 100, 1, 1).to(device)
            fake_images = self.shared_layers.generator(noise)
            fake_loss = self.criterion(self.shared_layers.discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer.step()

            # Train Generator
            self.optimizer.zero_grad()
            g_loss = self.criterion(self.shared_layers.discriminator(fake_images), real_labels)
            g_loss.backward()
            self.optimizer.step()
            # print(d_loss.item())
            # print(g_loss.item())
            batch_loss = d_loss + g_loss

        else:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            # Non-GAN Training Logic
            self.shared_layers.train(True)
            if self.specific_layers:
                self.specific_layers.train(True)
                output_batch = self.specific_layers(self.shared_layers(input_batch))
            else:
                output_batch = self.shared_layers(input_batch)
            self.optimizer.zero_grad()
            batch_loss = self.criterion(output_batch, label_batch)
            if not torch.isfinite(batch_loss):
                raise RuntimeError(f"Invalid loss value detected: {batch_loss.item()}")
            batch_loss.backward()
            self.optimizer.step()

        return batch_loss.item()

    def test_model(self, input_batch):
        self.shared_layers.train(False)
        with torch.no_grad():
            if self.specific_layers:
                output_batch = self.specific_layers(self.shared_layers(input_batch))
            else:
                output_batch = self.shared_layers(input_batch)
        self.shared_layers.train(True)
        return output_batch

    def update_model(self, new_shared_layers):
        self.shared_layers.load_state_dict(new_shared_layers)


def initialize_model(args, device):
    global shared_layers, specific_layers
    criterion = nn.CrossEntropyLoss()
    if args.mtl_model:
        print('Using different task specific layer for each user')
        if args.dataset == 'cifar10':
            if args.model == 'cnn_complex':
                shared_layers = cifar_cnn_3conv_shared(input_channels=3)
                feature_out_dim = shared_layers.feature_out_dim()
                specific_layers = cifar_cnn_3conv_specific(input_channels=feature_out_dim,
                                                           output_channels=10)
            else:
                raise ValueError('Model not implemented for CIFAR-10')
        else:
            raise ValueError('The dataset is not implemented for mtl yet')
        if args.cuda:
            shared_layers = shared_layers.cuda(device)
            specific_layers = specific_layers.cuda(device)
    elif args.global_model:
        print('Using same global model for all users')
        if args.dataset == 'cifar10' or args.dataset == 'cinic10':
            if args.model == 'cnn_complex':
                shared_layers = cifar_cnn_3conv(input_channels=3, output_channels=10)
                specific_layers = None
            elif args.model == 'resnet18':
                shared_layers = ResNet18()
                specific_layers = None
            elif args.model == 'resnet18_YWX':
                shared_layers = ResNet18_YWX()
                specific_layers = None
            else:
                raise ValueError('Model not implemented for CIFAR-10')
        elif args.dataset == 'femnist':
            if args.model == 'lenet':
                shared_layers = mnist_lenet(input_channels=1, output_channels=62)
                specific_layers = None
            elif args.model == 'logistic':
                shared_layers = LogisticRegression_MNIST(input_dim=1, output_dim=62)
                specific_layers = None
            elif args.model == 'cnn':
                shared_layers = mnist_cnn(input_channels=1, output_channels=62)
                specific_layers = None
            else:
                raise ValueError('Model not implemented for FEMNIST')
        elif args.dataset == 'mnist':
            if args.model == 'lenet':
                shared_layers = mnist_lenet(input_channels=1, output_channels=10)
                specific_layers = None
            elif args.model == 'logistic':
                shared_layers = LogisticRegression_MNIST(input_dim=1, output_dim=10)
                specific_layers = None
            elif args.model == 'cnn':
                shared_layers = mnist_cnn(input_channels=1, output_channels=10)
                specific_layers = None
            else:
                raise ValueError('Model not implemented for MNIST')
        elif args.dataset == 'synthetic':
            if args.model == 'logistic':
                shared_layers = LogisticRegression_SYNTHETIC(args.dimension, args.num_class)
                specific_layers = None
        elif args.dataset == 'shakespare':
            if args.model == 'rnn':
                shared_layers = RNNModel()
                specific_layers = None
            elif args.model == 'lstm':
                # shared_layers = LSTMModel()
                specific_layers = None
        elif args.dataset == 'celeba':
            if args.model == 'gan':
                shared_layers = GAN(Generator(), Discriminator())
                criterion = nn.BCELoss()
                specific_layers = None
        else:
            raise ValueError('The dataset is not implemented for mtl yet')
        if args.cuda:
            shared_layers = shared_layers.cuda(device)
    else:
        raise ValueError('Wrong input for the --mtl_model and --global_model, only one is valid')
    model = MTL_Model(shared_layers=shared_layers,
                      specific_layers=specific_layers,
                      learning_rate=args.lr,
                      lr_decay=args.lr_decay,
                      lr_decay_epoch=args.lr_decay_epoch,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      criterion = criterion)
    return model


def main():
    """
    For test this part
    --dataset: cifar-10
    --model: cnn_tutorial
    --lr  = 0.001
    --momentum = 0.9
    cpu only!
    check(14th/July/2019)
    :return:
    """
    args = args_parser()
    device = 'cpu'
    # build dataset for testing
    model = initialize_model(args, device)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    parent_dir = dirname(dirname(abspath(__file__)))
    data_path = join(parent_dir, 'data', 'cifar10')
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    for epoch in tqdm(range(350)):  # loop over the dataset multiple times
        model.step_lr_scheduler(epoch)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            loss = model.optimize_model(input_batch=inputs,
                                        label_batch=labels)

            # print statistics
            running_loss += loss
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model.test_model(input_batch=images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    main()
