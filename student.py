# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################


class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(6, 12, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(
            12, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(1536, 128, bias=True)
        self.cls = nn.Linear(128, 16, bias=True)
        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        x = self.cls(x)
        # TODO-BLOCK-END
        return x


def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    running_loss = 0.0
    num_correct = 0.0
    total_images = 0.0

    outputs = net(inputs)
    loss = criterion(outputs, labels.squeeze())
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    num_correct += torch.sum(preds == labels.data.reshape(-1))
    total_images += labels.data.numpy().size

    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images


#########################################################
###               DATA AUGMENTATION
#########################################################


class Shift(object):
    """
    Shifts input image by random x amount between [-max_shift, max_shift]
      and separate random y amount between [-max_shift, max_shift]. A positive
      shift in the x- and y- direction corresponds to shifting the image right
      and downwards, respectively.

      Inputs:
          max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """

    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        image = np.moveaxis(image, 0, -1)
        x = random.uniform(-(self.max_shift), self.max_shift)
        y = random.uniform(-(self.max_shift), self.max_shift)

        transMx = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, transMx, (W, H))
        image = np.moveaxis(image, -1, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        contrast = random.uniform(self.min_contrast, self.max_contrast)
        a = contrast
        b = 1 - contrast
        mean = image.mean(axis=0)
        image = a * image + b * mean
        # from oh
        image = np.minimum(image, 1)
        image = np.maximum(image, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """

    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        image = np.moveaxis(image, 0, -1)
        angle = random.uniform(-self.max_angle, self.max_angle)
        transMx = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
        image = cv2.warpAffine(image, transMx, (W, H))
        image = np.moveaxis(image, -1, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        image = np.moveaxis(image, 0, -1)
        if random.random() < self.p:
            image = cv2.flip(image, 1)
        image = np.moveaxis(image, -1, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


#########################################################
###             STUDENT MODEL
#########################################################


def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123.0 / 255.0, 116.0 / 255.0, 97.0 / 255.0]
    dataset_stds = [54.0 / 255.0, 53.0 / 255.0, 52.0 / 255.0]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer


class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN

        # TODO-BLOCK-END
        return x


#########################################################
###             ADVERSARIAL IMAGES
#########################################################


def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
    loss = criterion(output, label)
    net.zero_grad()
    loss.backward()

    sign_data_grad = img.grad.data.sign()
    noise = epsilon * sign_data_grad
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = img + noise
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # TODO-BLOCK-END

    return perturbed_image, noise
