import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms
# For splitting dataset object
from torch.utils.data import random_split

# To train the model, we want the images as one long vector.
dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

#Testing
image, label = dataset[0]
#print(image)
print(image.shape)
print(label)

print(type(image.shape))
print(type(label))

# Now that we have run the basic tests, lets flatten the pixels to feed to models.
pixels = image.flatten()
print(f"Number of pixels in a single image {pixels.shape}")
print(f"Number of images {len(dataset)}")

# Lets get the number of unique pixel values that our model has available to predict
# There are 256 possible grayscale values that our model can predict
num_possible_predictions = 256
vocab_size = num_possible_predictions

# Splitting the dataset into train and validation set. Since dataset is an object, we can't use a straghtforward slice to split the data.
n = int(0.9*len(dataset)) 
m = len(dataset) - n # Length of validation set
train_data, val_data = random_split(dataset, [n,m])

print("First image of the train set")
print(train_data[0])
# print("First image of the validation set")
# print(val_data[0])

# So lets now perform a very clean version of input-target on a single flattened vector from the training set.
first_image = train_data[0][0].flatten()
# print(first_image.flatten()[9]==first_image.flatten()[0])

# Preferably our context length should be 784 which is the length of flattened pixels of a 28x28 image
# However, for now, we shall restrict it to 128 before we run the sanity checks
# So this is max context length or block size

block_size = 128
x = first_image[:block_size]
y = first_image[1:block_size+1]

# Should print 0-127
# for pixel in range(block_size):
#     print(pixel)

for pixel in range(block_size):
    context = x[:pixel+1]
    target = y[pixel]
    print(f"when input is {context} the target: {target}")

# CREATE SPECIAL TOKEN FOR CLASSES

