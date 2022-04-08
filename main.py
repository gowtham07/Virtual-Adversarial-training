import argparse
import logging
import math
from test import test_cifar10, test_cifar100

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy
from vat import VATLoss

# set random seeds
seed = 42
torch.manual_seed(seed)
writer = SummaryWriter()
logger = logging.getLogger(__name__)


def test_accuracy_fn(test_logits, device, test_loader):
    labels_arr = torch.tensor([])
    labels_arr = labels_arr.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        labels_arr = torch.cat((labels_arr, labels), dim=0)
    return accuracy(test_logits, labels_arr)[0]


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(
            args, args.datapath
        )
        PATH = f"./model/model_lr:{args.lr}_epsilon_:{args.vat_eps}_num_labeleds_:{args.num_labeled}_train_batch:{args.train_batch}_model_depth:{args.model_depth}_cifar10.pt"
        Path2 = f"./model/model_lr:{args.lr}_epsilon_:{args.vat_eps}_num_labeleds_:{args.num_labeled}_train_batch:{args.train_batch}_model_depth:{args.model_depth}_cifar10_best_model.pt"
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(
            args, args.datapath
        )
        PATH = f"./model/model_lr:{args.lr}_epsilon_:{args.vat_eps}_num_labeleds_:{args.num_labeled}_train_batch:{args.train_batch}_model_depth:{args.model_depth}_cifar100.pt"
        Path2 = f"./model/model_lr:{args.lr}_epsilon_:{args.vat_eps}_num_labeleds_:{args.num_labeled}_train_batch:{args.train_batch}_model_depth:{args.model_depth}_cifar100_best_model.pt"
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unlabeled_dataset_train, unlabeled_dataset_validate = train_test_split(
        unlabeled_dataset, test_size=0.02, random_state=84
    )
    labeled_loader = iter(
        DataLoader(
            labeled_dataset,
            batch_size=args.train_batch,
            shuffle=True,
            num_workers=args.num_workers,
        )
    )
    unlabeled_loader = iter(
        DataLoader(
            unlabeled_dataset_train,
            batch_size=args.unlabelled_batch,
            shuffle=True,
            num_workers=args.num_workers,
        )
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        unlabeled_dataset_validate,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = WideResNet(
        args.model_depth, args.num_classes, widen_factor=args.model_width
    )
    model = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    lr_rate = args.lr
    loss_var = []
    best_val_acc = 0

    if args.train_mode:
        for epoch in range(args.epoch):
            for i in range(args.iter_per_epoch):
                try:
                    x_l, y_l = next(labeled_loader)
                except StopIteration:
                    labeled_loader = iter(
                        DataLoader(
                            labeled_dataset,
                            batch_size=args.train_batch,
                            shuffle=True,
                            num_workers=args.num_workers,
                        )
                    )
                    x_l, y_l = next(labeled_loader)
                x_l, y_l = x_l.to(device), y_l.to(device)

                try:
                    x_ul, _ = next(unlabeled_loader)
                except StopIteration:
                    unlabeled_loader = iter(
                        DataLoader(
                            unlabeled_dataset,
                            batch_size=args.train_batch,
                            shuffle=True,
                            num_workers=args.num_workers,
                        )
                    )
                    x_ul, _ = next(unlabeled_loader)

                x_ul = x_ul.to(device)

                ####################################################################
                # TODO: SUPPLY you code
                ####################################################################
                y_l = y_l.long()
                vatloss = VATLoss(args)
                vtloss = vatloss(model, x_ul)
                optimizer.zero_grad()
                outputs = model(x_l)
                org_loss = criterion(outputs, y_l)
                loss_var.append(vtloss)
                logger.info("kl-loss: {:.2f}".format(vtloss))
                loss = org_loss + args.alpha * vtloss
                loss.backward()
                optimizer.step()

            print(
                "[%d/%d] loss: %.3f, accuracy: %.3f"
                % (i, epoch, loss.item(), accuracy(outputs, y_l)[0])
            )

            print("loss-{0} vatloss: {1}".format(org_loss, vtloss))
            train_acc = accuracy(outputs, y_l)[0]
            train_loss = loss.item()
            writer.add_scalars(
                f"model_lr:{lr_rate}_num_labeleds_:{args.num_labeled}_epsilon_:{args.vat_eps}_train_batch:{args.train_batch}_model_depth:{args.model_depth}/Loss",
                {"train": train_loss},
                epoch,
            )
            writer.add_scalars(
                f"model_lr:{lr_rate}_num_labeleds_:{args.num_labeled}_epsilon_:{args.vat_eps}_train_batch:{args.train_batch}_model_depth:{args.model_depth}/Accuracy",
                {"train": train_acc},
                epoch,
            )
            model.eval()
            predicted_arr = torch.tensor([], device=device)
            labels_arr = torch.tensor([], device=device)
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    predicted_arr = torch.cat((predicted_arr, outputs), dim=0)
                    labels_arr = torch.cat((labels_arr, labels), dim=0)

            print(
                "Accuracy of the network on the validation images: %.3f"
                % (accuracy(predicted_arr, labels_arr)[0])
            )

            val_acc = accuracy(predicted_arr, labels_arr)[0]
            val_loss = loss.item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model, Path2)
                print("best model with val acc" + str(val_acc) + "is saved")

            writer.add_scalars(
                f"model_lr:{lr_rate}_num_labeleds_:{args.num_labeled}_epsilon_:{args.vat_eps}_train_batch:{args.train_batch}_model_depth:{args.model_depth}/Loss",
                {"validation": val_loss},
                epoch,
            )
            writer.add_scalars(
                f"model_lr:{lr_rate}_num_labeleds_:{args.num_labeled}_epsilon_:{args.vat_eps}_train_batch:{args.train_batch}_model_depth:{args.model_depth}/Accuracy",
                {"validation": val_acc},
                epoch,
            )
            model.train()
            torch.cuda.empty_cache()
        writer.add_hparams(
            {
                "learning_rate": lr_rate,
                "num_labeleds": args.num_labeled,
                "epsilon": args.vat_eps,
                "train_batch": args.train_batch,
                "model_depth": args.model_depth,
            },
            {
                "hp/t_loss": train_loss,
                "hp/v_loss": val_loss,
                "hp/t_accuracy": train_acc,
                "hp/v_accuracy": val_acc,
            },
        )
        torch.save(
            {
                "model_depth": args.model_depth,
                "num_classes": args.num_classes,
                "model_width": args.model_width,
                "device": device,
                "model_state_dict": model.state_dict(),
            },
            PATH,
        )

    # test
    else:
        model.eval()
        if args.dataset == "cifar10":

            test_logits = test_cifar10(test_dataset, filepath=args.run_model)
            acc = test_accuracy_fn(test_logits, device, test_loader)
            print("Accuracy of the network on the test images: %.3f" % acc)
        elif args.dataset == "cifar100":
            test_logits = test_cifar100(test_dataset, filepath=args.run_model)
            acc = test_accuracy_fn(test_logits, device, test_loader)
            print("Accuracy of the network on the test images: %.3f" % acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch"
    )
    parser.add_argument(
        "--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100"]
    )
    parser.add_argument(
        "--datapath",
        default="./data/",
        type=str,
        help="Path to the CIFAR-10/100 dataset",
    )
    parser.add_argument(
        "--num-labeled", type=int, default=2500, help="Total number of labeled samples"
    )
    parser.add_argument(
        "--lr", default=0.001, type=float, help="The initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="Optimizer momentum"
    )
    parser.add_argument("--wd", default=0.0005, type=float, help="Weight decay")
    parser.add_argument(
        "--expand-labels", action="store_true", help="expand labels to fit eval steps"
    )
    parser.add_argument("--train-batch", default=32, type=int, help="train batchsize")
    parser.add_argument("--test-batch", default=64, type=int, help="train batchsize")
    parser.add_argument(
        "--unlabelled-batch", default=128, type=int, help="train batchsize"
    )
    parser.add_argument(
        "--total-iter",
        default=1600 * 50,
        type=int,
        help="total number of iterations to run",
    )
    parser.add_argument(
        "--iter-per-epoch",
        default=1600,
        type=int,
        help="Number of iterations to run per epoch",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="Number of workers to launch during training",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        metavar="ALPHA",
        help="regularization coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--dataout",
        type=str,
        default="./path/to/output/",
        help="Path to save log files",
    )
    parser.add_argument(
        "--model-depth", type=int, default=34, help="model depth for wide resnet"
    )
    parser.add_argument(
        "--model-width", type=int, default=2, help="model width for wide resnet"
    )
    parser.add_argument("--vat-xi", default=0.5, type=float, help="VAT xi parameter")
    parser.add_argument(
        "--vat-eps", default=5.0, type=float, help="VAT epsilon parameter"
    )
    parser.add_argument(
        "--vat-iter", default=1, type=int, help="VAT iteration parameter"
    )
    parser.add_argument("--run-model", default="", type=str, help="model_file_name")
    parser.add_argument("--train-mode", default=1, type=int)

    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments

    args = parser.parse_args()
    main(args)
