Implementation of Takeru Miyato et al. “Virtual adversarial training: a regularization method for supervised and semi-supervised learning”. In: IEEE transactions on pattern analysis and machine intelligence 41.8 (2018), pp. 1979–1993.

To test the provided models in the model folder for CIFAR-10 250 and 4000 labels and for CIFAR-100 2500 and 10000 labels, please provide the argument for main.py of appropriate model file name and the second argument of flag to set in train mode.

Along with the third argument of which dataset to be choosen("cifar10"or"cifar100")

(eg main.py --run-model "./model/model_lr_0.001_epsilon__2.0_num_labeleds__250_train_batch_32_model_depth_34_cifar10.pt" --train-mode 0 --dataset "cifar10")


Models saved:

model_lr_0.001_epsilon__5.0_num_labeleds__10000_train_batch_32_model_depth_28_cifar100.pt
model_lr_0.001_epsilon__5.0_num_labeleds__2500_train_batch_32_model_depth_34_cifar100.pt
model_lr_0.001_epsilon__2.0_num_labeleds__4000_train_batch_32_model_depth_28_cifar10.pt
model_lr_0.001_epsilon__2.0_num_labeleds__250_train_batch_32_model_depth_34_cifar10.pt