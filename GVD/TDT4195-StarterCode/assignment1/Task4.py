import torch
import matplotlib.pyplot as plt
import utils
import dataloaders
import torchvision
from trainer import Trainer

torch.random.manual_seed(0)


class SingleLayerModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28 * 28
        # Number of classes in the MNIST dataset
        num_classes = 10

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28 * 28)
        out = self.classifier(x)
        return out


class DoubleLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28 * 28
        # Number of nodes second layer
        num_hidden_layer = 64
        # Number of classes in the MNIST dataset
        num_classes = 10

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, num_hidden_layer),
        )
        self.classifier2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden_layer, num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28 * 28)
        x = self.classifier(x)
        out = self.classifier2(x)
        return out


# ### Hyperparameters & Loss function

# Hyperparameters
batch_size = 64
learning_rate = 0.0192
num_epochs = 5

# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Model1 definition
model1 = SingleLayerModel()

# Model2 definition
model2 = DoubleLayerModel()

# Define optimizer (Stochastic Gradient Descent)
optimizer1 = torch.optim.SGD(model1.parameters(),
                             lr=learning_rate)

optimizer2 = torch.optim.SGD(model2.parameters(),
                             lr=learning_rate)
# Weight
weight1 = next(model1.classifier.children()).weight.data
print(len(weight1[0]))

weight2 = next(model2.classifier.children()).weight.data
print(len(weight2[0]))

image_transform1 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
image_transform2 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.25,)),
])

dataloader_train1, dataloader_val1 = dataloaders.load_dataset(batch_size, image_transform=image_transform1)
dataloader_train2, dataloader_val2 = dataloaders.load_dataset(batch_size, image_transform=image_transform2)


trainer1 = Trainer(
    model=model1,
    dataloader_train=dataloader_train1,
    dataloader_val=dataloader_val1,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer1,
)
trainer2 = Trainer(
    model=model1,
    dataloader_train=dataloader_train2,
    dataloader_val=dataloader_val2,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer1,
)

train_loss_dict1, val_loss_dict1 = trainer1.train(num_epochs)
train_loss_dict2, val_loss_dict2 = trainer2.train(num_epochs)

# Plot loss

utils.plot_loss(train_loss_dict1, label="Train Loss")
utils.plot_loss(val_loss_dict1, label="Test Loss")
utils.plot_loss(train_loss_dict2, label="Train Loss normalized")
utils.plot_loss(val_loss_dict2, label="Test Loss normalized")
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Number of Images Seen")
plt.ylabel("Cross Entropy Loss")
plt.savefig("training_loss.png")

plt.show()

torch.save(model1.state_dict(), "saved_model1.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_val1, model1, loss_function)
print(f"Final Test Cross Entropy Loss: {final_loss}. Final Test accuracy: {final_acc}")

torch.save(model2.state_dict(), "saved_model2.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_val2, model1, loss_function)
print(f"Final Test Cross Entropy Loss: {final_loss}. Final Test accuracy: {final_acc}")

import numpy as np

for i in range(len(weight1)):
    fig, ax = plt.subplots()

    im = ax.imshow(np.reshape(weight1[i], (28, 28)), cmap='gray')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xticks(np.arange(28))
    ax.set_yticks(np.arange(28))
    ax.set_xticklabels(range(28))
    ax.set_yticklabels(range(28))
    ax.set_title("weight " + str(i))
    fig.tight_layout()
    plt.savefig("Weight" + str(i) + ".png")
    plt.show()
