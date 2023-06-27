from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
from glob import glob
from torch import load, cat

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.pt')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, extensions=None, is_valid_file=None, features=None):
    images = []
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    if features:
        files = glob(os.path.join(dir, "features_scale_1", "*", "*", "*"))
    else:
        files = glob(os.path.join(dir, "images", "*", "*", "*"))

    for path in sorted(files):
        if is_valid_file(path):
            target = 0
            if "/morphs/" in path:
                target=1
            item = (path, target)
            images.append(item)
    return images

class DatasetFolder(VisionDataset):

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, args=None, features=False):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        # write one generic and few dataset-specific functions make_dataset()
        self.features=features
        samples = make_dataset(self.root, extensions, is_valid_file, features=self.features)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.features=features
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample=None
            if self.features:
                # load feature tensors
                sample = cat((load(path).unsqueeze(0), load(path.replace("/features_scale_1/", "/features_scale_2/"))), dim=0)
            else:
                sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
        except:
            print("Error: {} was not loaded ".format(path))

        return sample, target, path

    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, args=None, features=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          args=args,
                                          features=features)
        self.imgs = self.samples