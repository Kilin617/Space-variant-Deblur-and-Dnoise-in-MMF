import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2


class SingleDataset(BaseDataset):
    def __init__(self):
        super().__init__()

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5),
                                               (0.5))]

        self.transform = transforms.Compose(transform_list)

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        A_img = Image.open(A_path)
        if len(A_img.size) == 2:
            A_img = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
        elif len(A_img.size) ==3:
            A_img = Image.open(A_path).convert('RGB')
        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
