import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'blur')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'clear')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5),
                                               (0.5))]

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        B_path = self.B_paths[index % self.A_size]
        A_img = Image.open(A_path)
        if len(A_img.size) == 2:
            A_img = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
        elif len(A_img.size) == 3:
            A_img = Image.open(A_path).convert('RGB')

        B_img = Image.open(B_path).convert('RGB')
        if len(B_img.size) == 2:
            B_img = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE)
        elif len(A_img.size) == 3:
            B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
