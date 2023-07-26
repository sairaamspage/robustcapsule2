import torch.nn as nn
import torch.nn.functional as F
import torch
from constants import *
import torch.optim as optim 
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os
import time
import math
import sys
sys.path.append('./')

from model import *
from data_loader import *

best_accry = 0.0
loss_criterion = nn.CrossEntropyLoss()

def get_model(desc):
    if desc == 'GcnnSovnet':
       model = GcnnSovnet()
       model = nn.DataParallel(model).to(DEVICE)
       return model      

def onehot(tensor, num_classes=100):
    return torch.eye(num_classes).index_select(dim=0, index=tensor)# One-hot encode

def check_point(model,optimizer,save_file,lr_scheduler,loss,epoch,accuracy):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            'accuracy_model': accuracy  
            }, save_file)

def train_test(num_epochs,train_loaders_desc,test_loaders_desc,trial_num):
    for (train_loader, train_loader_desc) in train_loaders_desc:
        model = nn.DataParallel(ResidualSovnet().to(DEVICE))#get_model(baseline)
        #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(pytorch_total_params)
        optimizer = optim.AdamW(model.parameters())#, weight_decay=2e-4)
        #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        #step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,200])
        fast_convergence_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, 2e-3, epochs=150, steps_per_epoch=len(train_loader))
        #optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)
        #exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch : 0.99**(epoch))
        save_file = SAVEPATH+train_loader_desc+'_'+'train_model_trial_'+str(trial_num)+'.pth'
        best_save_file = SAVEPATH+train_loader_desc+'_'+'highest_accuracy_model_trial_'+str(trial_num)+'.pth'
        train(model,optimizer,fast_convergence_lr_scheduler,train_loader,test_loaders_desc,save_file,best_save_file,num_epochs,0.5,0.9,0.1, train_loader_desc)
         
def train(model,optimizer,lr_scheduler,train_loader,test_loaders_desc,save_file,best_save_file,num_epochs,lambda_,positive_margin,negative_margin,train_loader_desc):
    for epoch in range(num_epochs):
        global best_accry   
        train_total = 0.0
        train_correct = 0.0
        train_loss_aggregate = 0.0
        loss_history = []
        model.train()
        iteration = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            iteration += 1
            #target = onehot(target) 
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            activation = model(data)
            #activations = get_activations(capsules)
            loss = loss_criterion(activation, target)#margin_loss(activation,target,lambda_,positive_margin, negative_margin)
            train_loss_aggregate += loss.item() 
            loss.backward()
            optimizer.step()
            predictions = activation.max(dim=1)[1]#get_predictions(activation)
            train_total += len(data)
            train_correct += (predictions == target).sum()#(predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum()
            train_accuracy = float(train_correct)/float(train_total)
            lr_scheduler.step()
            if batch_idx % 100 == 0:
               #print(model_name)
               #print(train_loader_desc)
               print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}'.format(
                     epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss, train_accuracy))
               #print('loss {}'.format(loss))
               #print('margin loss {}'.format(margin_loss))
               #print('reconstruction loss {}'.format(reconstruction_loss))
               #print('train_accuracy {}'.format(train_accuracy))
        #loss_history.append(train_loss_aggregate/len(train_loader))
        print('train_accuracy ', train_accuracy)
        print('testing')
        if epoch % 1 == 0:
           with torch.no_grad():
               test_accuracy_list, test_loss_list = test(model,test_loaders_desc,epoch,train_loader_desc)
               print(test_accuracy_list)
               if test_accuracy_list[0] >= best_accry:
                  best_accry = test_accuracy_list[0]
                       
        if epoch%1 == 0:
           check_point(model,optimizer,save_file,lr_scheduler,train_loss_aggregate/len(train_loader),epoch,train_accuracy)
           check_point(model,optimizer,best_save_file,lr_scheduler,test_loss_list,epoch,test_accuracy_list)

def margin_loss(class_activations,target,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    batch_size = class_activations.size(0)
    left = F.relu(positive_margin - class_activations).view(batch_size, -1) ** 2
    right = F.relu(class_activations - negative_margin).view(batch_size, -1) ** 2
    margin_loss = target * left + lambda_ *(1-target)*right
    margin_loss = margin_loss.sum(dim=1).mean()
    return margin_loss

def test(model,test_loaders_desc,epoch,train_loader_desc,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    test_accuracy_list = []
    test_loss_list = []
    model.eval()
    #iteration = 0
    for (test_loader,test_loader_desc) in test_loaders_desc:
         #iteration += 1
         test_loss = 0.0
         #margin_loss = 0.0
         #reconstruction_loss = 0.0
         test_correct = 0.0
         test_total = 0.0 
         iterations = 0
         for trial in range(NUM_OF_TEST_TRIALS):
             for data, target in test_loader:
                 iterations += 1
                 #target = onehot(target)
                 data, target = data.to(DEVICE), target.to(DEVICE)
                 capsules = model(data)
                 #activations = get_activations(capsules)
                 predictions = capsules.max(dim=1)[1]#get_predictions(capsules)#activations)
                 margin_loss_temp = loss_criterion(capsules, target)#margin_loss(capsules, target,lambda_,positive_margin, negative_margin)# sum up batch loss
                 #margin_loss += margin_loss_temp
                 #reconstruction_loss += reconstruction_loss_temp
                 test_loss += margin_loss_temp#test_loss_temp
                 test_correct += float((predictions == target).sum().item())#float((predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum().item())
                 test_total += len(data)
         test_loss /= float(test_total)
         test_correct /= float(test_total)
         print(test_loader_desc) 
         print('\nTest set: Average loss: {:.4f}, Accuracy: {})\n'.format(test_loss, test_correct))
         test_loss_list.append(test_loss.item())
         test_accuracy_list.append(test_correct)
    return test_accuracy_list, test_loss_list

def main_loop():
    num_epochs = 150
    train_translation_rotation_list = [((0.065,0.065),180)]#,((0.065,0.065),90),((0,0),0)]
    test_translation_rotation_list = [((0,0),0),((0.065,0.065),30),((0.065,0.065),60),((0.065,0.065),90),((0.065,0.065),180)]
    train_loaders_desc, test_loaders_desc = get_loaders_cifar100(train_translation_rotation_list,test_translation_rotation_list,BATCHSIZE) 
    train_test(num_epochs,train_loaders_desc,test_loaders_desc,0)

if __name__ == '__main__':
   main_loop()





