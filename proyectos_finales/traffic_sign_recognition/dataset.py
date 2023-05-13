from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def get_dataloaders():
    """Returns train and validation dataloaders for the traffic sign recognition dataset"""
    file_path = Path(__file__).parent.absolute()
    root_path = file_path / "data/crop_dataset/crop_dataset/"

    # https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html    
    dataset = ImageFolder(root=root_path,
                          transform=get_transform())

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=64,
                                shuffle=False)
    return train_dataloader, val_dataloader


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ]
    )


def visualize_data():
    train_dataloader, val_dataloader = get_dataloaders()

    # Visualize some training images
    plt.figure(figsize=(8, 8))
    for data, target in train_dataloader:
        img_grid = make_grid(data)
        plt.axis("off")
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()
        print(data.shape)
        print(target.shape)
        break

    # Visualize some validation images with labels
    plt.figure(figsize=(8, 8))
    for data, target in val_dataloader:
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.axis("off")
            plt.imshow(data[i].permute(1, 2, 0))
            plt.title(target[i].item())
        plt.show()
        break


def main():
    visualize_data()


if __name__ == "__main__":
    main()
