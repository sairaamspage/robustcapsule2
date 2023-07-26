import torch

DATAPATH = "../../../cifar100/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
SAVEPATH = "./checkpoints/"
BATCHSIZE = 64
NUM_OF_TRAIN_TRIALS = 1
NUM_OF_TEST_TRIALS = 1


