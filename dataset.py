from torch.utils.data import Dataset
import os
from skimage import io

class IntelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # get names for all classes
        self.classes = sorted(os.listdir(self.root_dir))

        # make index for all classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.images = self._load_images()

    def _load_images(self):
        images = []

        # loop through all classes name
        for class_name in self.classes:

            # form the path to the folder with class pictures by concatenating root directory with class name
            class_dir = os.path.join(self.root_dir, class_name)

            # check if class_dir is a directory
            if not os.path.isdir(class_dir):
                continue

            # loop through all images from class class_name
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)

                # chack if file_path is file
                if os.path.isfile(file_path):

                    # append to images (file_path, file index from class class_name)
                    images.append((file_path, self.class_to_idx[class_name]))

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # get path to image and label for this image
        image_path, label = self.images[index]

        # open image and convert to RGB format
        image = io.imread(image_path) / 255.0

        # if transform not exist than apply this transform to image
        if self.transform is not None:
            image = self.transform(image)

        return (image, label)