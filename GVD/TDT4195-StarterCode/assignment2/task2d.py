'''
# Visualizing filters in Pytorch
'''

'''
### Imports
'''

import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np

image = Image.open("images/zebra.jpg")
plt.imshow(image)
print("Image shape:", image.size)

'''
### Model definition
In this example we will use a pre-trained ResNet50 network. ResNet-50 is a fully-convolutional neural network that excels at image classification.

'''

model = torchvision.models.resnet50(pretrained=True)
print(model)

'''
### First convolution layer
In this task we are interested in visualizing the first convolutional layer. This can be retrieved by the following code block:

We can see that it has 64 filters/kernels in the layer. Each kernel is a $7 \times 7$ filter, that takes an RGB image as input
'''

first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

'''
### Activation from first layer
We can retrieve the activation from the first layer by doing a forward pass throught this conv layer.

However, first we need to resize, and normalize the image with the mean and standard deviation that they used to originally train this network.
'''

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)

'''
### Visualize filters & Activation
'''

# tip
# To transform a weight to a numpy array, you can use to function
def torch_image_to_numpy(image: torch.Tensor):
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image
    


### START YOUR CODE HERE ### (You can change anything inside this block)
# plt.subplot is a nice function to use for this task!
indices = [5, 8, 19, 22, 34]

fig, axs = plt.subplots(2, 5)

#plt.subplot(1, 5, 1)
#plt.imshow(torch_image_to_numpy(first_conv_layer.weight[5].data))
axs[0, 0].imshow(torch_image_to_numpy(first_conv_layer.weight[5].data))
#plt.subplot(1, 5, 2)
#plt.imshow(torch_image_to_numpy(first_conv_layer.weight[8].data))
axs[0, 1].imshow(torch_image_to_numpy(first_conv_layer.weight[8].data))
#plt.subplot(1, 5, 3)
#plt.imshow(torch_image_to_numpy(first_conv_layer.weight[19].data))
axs[0, 2].imshow(torch_image_to_numpy(first_conv_layer.weight[19].data))

#plt.subplot(1, 5, 4)
#plt.imshow(torch_image_to_numpy(first_conv_layer.weight[22].data))
axs[0, 3].imshow(torch_image_to_numpy(first_conv_layer.weight[22].data))

#plt.subplot(1, 5, 5)
#plt.imshow(torch_image_to_numpy(first_conv_layer.weight[34].data))
axs[0, 4].imshow(torch_image_to_numpy(first_conv_layer.weight[34].data))


#plt.subplot(2, 5, 1)
#plt.imshow(activation[0, 5].data, cmap="gray")
axs[1, 0].imshow(activation[0, 5].data, cmap="gray")

#plt.subplot(2, 5, 2)
#plt.imshow(activation[0, 8].data, cmap="gray")
axs[1, 1].imshow(activation[0, 8].data, cmap="gray")

#plt.subplot(2, 5, 3)
#plt.imshow(activation[0, 19].data, cmap="gray")
axs[1, 2].imshow(activation[0, 19].data, cmap="gray")

#plt.subplot(2, 5, 4)
#plt.imshow(activation[0, 22].data, cmap="gray")
axs[1, 3].imshow(activation[0, 22].data, cmap="gray")

#plt.subplot(2, 5, 5)
#plt.imshow(activation[0, 34].data, cmap="gray")
axs[1, 4].imshow(activation[0, 34].data, cmap="gray")

plt.figure(figsize=(20, 4))

plt.show()
### END YOUR CODE HERE ###

