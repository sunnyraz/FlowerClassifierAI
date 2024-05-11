import argparse
import json
import logging
import argparse
import torch
from PIL import Image
import numpy as np
import torch
import logging
from torch import nn
import torch.nn.functional as F
from torchvision import models

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.resize((256, 256))

    # Crop the center 224x224 portion of the image
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Normalize the image
    image_means = np.array([0.485, 0.456, 0.406])
    image_stds = np.array([0.229, 0.224, 0.225])
    normalized_image_array = (image_array / 255 - image_means) / image_stds

    # Reorder the color channels to the first dimension
    transformed_image_array = np.transpose(normalized_image_array, (2, 0, 1))

    return transformed_image_array


def load_checkpoint(checkpoint_path, device):
    """loads checkpoint and rebuilds the model"""

    checkpoint = torch.load(checkpoint_path)

    arch = checkpoint['arch']

    if arch != 'vgg16':
        raise ("Prediction cannot be initiated other than vgg16 model")

    model = models.vgg16(pretrained=False)

    classifier = nn.Sequential(nn.Linear(25088, 204),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(204, 102),
                               nn.LogSoftmax(dim=1)
                               )

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])

    class_to_idx = checkpoint['class_to_idx']
    model = model.to(device)

    return model, class_to_idx


def predict(image_path, checkpoint_path, top_k, gpu=False):
    """Predict image name and class probability"""

    if gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info("GPU enabled for model prediction")
    else:
        device = "cpu"

    model, class_to_idx = load_checkpoint(checkpoint_path, device)

    model.eval()

    img = Image.open(image_path)
    image = process_image(img)

    pre_processed_image = torch.from_numpy(image)
    pre_processed_image = torch.unsqueeze(
        pre_processed_image, 0).to(device).float()

    with torch.no_grad():
        log_ps = model(pre_processed_image)
        out_ps = torch.exp(log_ps)
        top_k, p_class = out_ps.topk(top_k, dim=1)

    classes = []
    reversed_class_to_idx = {value: key for key, value in class_to_idx.items()}
    for index in p_class.tolist()[0]:
        classes.append(reversed_class_to_idx[index])

    return top_k.tolist()[0], classes


def main():
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image with predict.py along with the probability of that name.')
    parser.add_argument('input', help='Path to the image')
    parser.add_argument('checkpoint', help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Return top K most likely classes (default: 1)')
    parser.add_argument('--category_names',
                        help='Path to the mapping of categories to real names (JSON file)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')

    args = parser.parse_args()

    probs, classes = predict(args.input, args.checkpoint, args.top_k, args.gpu)
    logger.info(probs)
    logger.info(classes)

    if args.category_names:
        with open(args.category_names, "r") as ft:
            cat_to_name = json.load(ft)
        flower_species = [cat_to_name[cl] for cl in classes]
        logger.info(flower_species)


if __name__ == '__main__':
    main()
