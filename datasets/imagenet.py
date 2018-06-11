import torch
from torchvision import transforms, datasets
import os


class ImageNet(datasets.ImageFolder):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ])

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        traindir = os.path.join(self.root, 'train')
        valdir = os.path.join(self.root, 'val')
        self.train = train
        transform = transform or self.preprocess()
        super(ImageNet, self).__init__(train and traindir or valdir,
                                       transform=transform, target_transform=target_transform)

    def preprocess(self):
        if self.train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Lighting(0.1, self.eigval, self.eigvec),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])


class Lighting(object):
    """Lighting noise (AlexNet-style PCA-based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.alpha = torch.Tensor(3).normal_(0, self.alphastd)

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        rgb = torch.mm(self.eigvec, torch.mul(self.alpha, self.eigval).view(3, 1))

        return img + rgb.view(3, 1, 1)
