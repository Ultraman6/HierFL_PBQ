from datasets.cifar_mnist import show_distribution, get_mnist, get_cifar10, get_femnist
from datasets.cinic10 import get_cinic10
from datasets.shakespare.shakespare import get_shakespare
from datasets.synthetic import get_synthetic


def get_dataset(dataset_root, dataset, args):
    # trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'cinic10':
        train_loaders, test_loaders, v_test_loader = get_cinic10(dataset_root, args)
    elif dataset == 'femnist':
        train_loaders, test_loaders, v_test_loader = get_femnist(dataset_root, args)
    elif dataset == 'synthetic':
        train_loaders, test_loaders, v_test_loader = get_synthetic(args)
    elif dataset == 'shakespare':
        train_loaders, test_loaders, v_test_loader = get_shakespare(dataset_root, args)
    elif dataset == 'CelebA':
        train_loaders, test_loaders, v_test_loader = get_CelebA(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_test_loader

def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ['mnist', 'cifar10', "femnist", "synthetic", "cinic10"]:
        train_loaders, test_loaders, v_test_loader = get_dataset(dataset_root=args.dataset_root,
                                                                                       dataset=args.dataset,
                                                                                       args = args)
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, test_loaders, share_data_edge, v_test_loader
