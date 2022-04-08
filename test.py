import torch
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_cifar10(test_dataset, filepath="./path/to/model.pth.tar"):
    """
    args:
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape
                [num_samples, 10]. Apply softmax to the logits

    Description:
        This function loads the model given in the filepath and returns the
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc)
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    """
    # TODO: SUPPLY the code for this function

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    if device.type == "cpu":
        checkpoint = torch.load(filepath, map_location="cpu")
    else:
        checkpoint = torch.load(filepath)

    model = WideResNet(
        checkpoint["model_depth"], checkpoint["num_classes"], checkpoint["model_width"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    predicted_arr = torch.tensor([])
    predicted_arr = predicted_arr.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_arr = torch.cat((predicted_arr, outputs), dim=0)

    return predicted_arr


def test_cifar100(test_dataset, filepath="./path/to/model.pth.tar"):
    """
    args:
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape
                [num_samples, 100]. Apply softmax to the logits

    Description:
        This function loads the model given in the filepath and returns the
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc)
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    """
    # TODO: SUPPLY the code for this function
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    if device.type == "cpu":
        checkpoint = torch.load(filepath, map_location="cpu")
    else:
        checkpoint = torch.load(filepath)
    model = WideResNet(
        checkpoint["model_depth"], checkpoint["num_classes"], checkpoint["model_width"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    predicted_arr = torch.tensor([])
    predicted_arr = predicted_arr.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_arr = torch.cat((predicted_arr, outputs), dim=0)

    return predicted_arr
