import os
import logging
import argparse
import torch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def preprocess_images(train_dir, valid_dir):
    """Preprocessing and splitting data for train, test and validation set"""

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=64, shuffle=True)

    class_to_idx = train_data.class_to_idx

    return trainloader, validloader, class_to_idx


def build_model(hidden_units):
    """
    Build Model Architecture
    """
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1)
                               )
    model.classifier = classifier
    return model


def save_checkpoint(model, class_to_idx, save_dir, arch, epochs):
    """
    Save the model checkpoint
    """
    if save_dir != "./checkpoint.pth" and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, "checkpoint.pth")

    model.class_to_idx = class_to_idx
    torch.save({
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epoch': epochs,
        'arch': arch
    }, save_dir)


def train(data_dir, save_dir=None, arch='vgg16', learning_rate=0.002, hidden_units=204, epochs=12, gpu=False):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    trainloader, validloader, class_to_idx = preprocess_images(
        train_dir, valid_dir)

    if gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info("GPU enabled for model training")
    else:
        device = "cpu"

    if arch != 'vgg16':
        raise ("Training cannot be initiated other than vgg16 model")

    model = build_model(hidden_units)
    model = model.to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    step = 0
    print_every = 5
    running_loss = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            step += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)

            loss = criterion(logps, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if step % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in validloader:

                        inputs, labels = inputs.to(
                            device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate Accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(
                            *top_class.shape)
                        accuracy += torch.mean(
                            equality.type(torch.FloatTensor)).item()

                logger.info(f"Epoch {epoch+1}/{epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Validation loss: {test_loss/len(validloader):.3f}.. "
                            f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    save_checkpoint(model, class_to_idx, save_dir, arch, epochs)


def main():
    parser = argparse.ArgumentParser(
        description='Train a new network on a data set with train.py')
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument(
        '--arch', choices=['vgg16'], default='vgg16', help='Choose architecture (default: vgg16)')
    parser.add_argument('--save_dir', dest="save_dir",
                        action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', type=float,
                        default=204, help='Set learning rate (default: 0.002)')
    parser.add_argument('--hidden_units', type=int, default=204,
                        help='Set number of hidden units (default: 204)')
    parser.add_argument('--epochs', type=int, default=12,
                        help='Set number of epochs (default: 12)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir

    train(data_dir, save_dir, args.arch,
          args.learning_rate, args.hidden_units, args.epochs, args.gpu)


if __name__ == '__main__':
    main()
