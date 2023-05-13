import torch.nn as nn
import torch
from pathlib import Path

file_path = Path(__file__).parent.absolute()

#TODO: Define la red neuronal


def main():
    net = Network(3, 43)
    print(net)
    torch.rand(1, 3, 32, 32)
    print(net(torch.rand(1, 3, 32, 32)))


if __name__ == "__main__":
    main()
