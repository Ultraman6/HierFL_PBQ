import os
import random
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from datasets.cifar_mnist import split_data


def get_celeba(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load CelebA dataset
    train = datasets.CelebA(root=dataset_root, split='train', target_type='attr', transform=transform, download=True)
    test = datasets.CelebA(root=dataset_root, split='test', target_type='attr', transform=transform, download=True)

    # Custom function to split data (implementation depends on your specific needs)
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)

    test_loaders = []
    if args.test_on_all_samples == 1:
        for i in range(args.num_clients):
            test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)
            test_loaders.append(test_loader)
    else:
        test_loaders = split_data(test, args, kwargs, is_shuffle=False)

    # Creating a validation subset from the test set
    test_set_size = len(test)
    subset_size = int(test_set_size * args.test_ratio)  # For example, retain 20% of the data
    indices = list(range(test_set_size))
    subset_indices = random.sample(indices, subset_size)
    subset = Subset(test, subset_indices)
    v_test_loader = DataLoader(subset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_test_loader