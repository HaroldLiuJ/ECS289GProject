import datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

from clip import clip


class COCODataset(Dataset):
    def __init__(self, img_processor, text_encoder, split=None):
        super(COCODataset, self).__init__()
        assert type(split) is str

        self.ds = self.load_dataset_from_huggingface(split)
        self.img_processor = img_processor
        self.text_encoder = text_encoder

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]
        caption = data['caption']
        image_path = data['image_path']

        image = self.img_processor(Image.open(image_path)).squeeze(0)
        text = self.text_encoder(caption).squeeze(0)

        return text, image


    def load_dataset_from_huggingface(self, split):
        COCO_DIR = "/home/xiao/projects/CLIP/data"
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, split=split)
        return ds

class CifarDataset(Dataset):
    def __init__(self, dataset_name, img_processor, text_encoder, split=None):
        super(CifarDataset, self).__init__()
        assert dataset_name in ['cifar10', 'cifar100']
        self.dataset_name = dataset_name
        self.ds = datasets.load_dataset(dataset_name, split=split)
        self.img_processor = img_processor
        self.text_encoder = text_encoder

        if dataset_name == 'cifar10':
            self.class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            self.class_labels = ['aquatic mammals', 'fish', ' flowers', 'food containers', 'fruit and vegetables',
                                 'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                                 'large man-made outdoor things', ' large natural outdoor scenes', 'large omnivores and herbivores',
                                 'medium mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                                 'vehicles 1', ' vehicles 2']

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]

        captions = ['an image of {}'.format(label) for label in self.class_labels]
        image= data['img']

        image = self.img_processor(image).squeeze(0)
        text = self.text_encoder(captions).squeeze(0)

        if self.dataset_name == 'cifar10':
            label = data['label']
        else:
            label = data['coarse_label']

        return text, image, label


class OxfordDataset(Dataset):
    def __init__(self, img_processor, text_encoder, split=None, dataset_name='pcuenq/oxford-pets'):
        super(OxfordDataset, self).__init__()
        assert dataset_name == 'pcuenq/oxford-pets'
        self.dataset_name = dataset_name
        self.ds = datasets.load_dataset(dataset_name, split=split)
        self.img_processor = img_processor
        self.text_encoder = text_encoder
        self.class_labels = [0, 1]


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]

        captions = ['a photo of a cat', 'a photo of a dog']
        image= data['image']

        image = self.img_processor(image).squeeze(0)
        text = self.text_encoder(captions).squeeze(0)

        label = 1 if data['dog'] else 0

        return text, image, label


def get_dataloader(dataset, batch_size):
    dataloader= DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # ds = COCODataset(preprocess, clip.tokenize, 'validation')
    #
    # dataloader = get_dataloader(ds, 16)

    # ds = CifarDataset('cifar10', preprocess, clip.tokenize, 'test')
    ds = OxfordDataset(preprocess, clip.tokenize, 'train')
    dataloader = get_dataloader(ds, 16)

    for text, image, label in dataloader:

        print(text.shape)
        print(image.shape)
        print(label)
        break