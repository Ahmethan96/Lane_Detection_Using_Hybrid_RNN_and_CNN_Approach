import torch
import argparse
import numpy as np

torch.cuda.empty_cache()
# globel param
# dataset setting
lab_height = 128
lab_channel = 1
data_loader = 4
class_numbers = 2
image_width = 256
image_height = 128
image_channel = 3
lab_width = 256


#class_weight = [0.02,1.02]
# class_weight =np.array(class_weight,float)
# weight
#class_weight= [0.02, 1.02]
#ibo = [0.02, 1.02]
#class_weight = [0.02, 1.02]
#class_weight = torch.stack(list([0.02, 1.02]))
#class_weight = tuple(map(torch.stack, zip([0.02, 1.02])))
#class_weight= torch.stack([0.02, 1.02], dim=0)
#class_weight= torch.tensor(ibo)
class_weights = [0.02, 1.02]

def arguments_setup():
    # Training settings
    model = argparse.ArgumentParser(description="UNet")
    model.add_argument('--model',type=str, default='UNet-ConvLSTM',help='( UNet-ConvLSTM | SegNet-ConvLSTM | UNet | SegNet | ') #model
    model.add_argument( '--batch-size',type=int, default=1, metavar='N',) # train batch size
    model.add_argument( '--test-batch-size',type=int, default=1, metavar='N')# test batch size
    model.add_argument( '--epochs',type=int, default=30, metavar='N')# number of epochs
    model.add_argument('--lr', type=float, default=0.01, metavar='LR')
    model.add_argument('--momentum', type=float, default=0.5, metavar='M') #SGD optimizer used
    model.add_argument('--cuda', action='store_true', default=True) #GPU used for training
    model.add_argument('--seed', type=int, default=1, metavar='S')
    model.add_argument('--log-interval', type=int, default=10, metavar='N')
    arguments = model.parse_args()
    return arguments
torch.cuda.empty_cache()