#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import dataset
import redishelper
import models

import sys
import time
import argparse


parser = argparse.ArgumentParser(description="Parse argument for training GoSGD with PyTorch")

# arguments for training data
parser.add_argument("-d","--dataset", default=".", help="Training/test data directory")
parser.add_argument("--classes",default=10,type=int,help="The number of classification")
parser.add_argument("--labels",default="0-9",help="The select labels")

# model related
parser.add_argument("--model",default="resnet18",help="The name of neural networks",choices=["resnet18",
"mobilenetv2","resnet34","resnet50","resnet101","resnet152","alexnet","alexnetimg8"])
parser.add_argument("--shape",default = "-1", nargs="+", type=int, help="The shape of training images")

# optimizer related
parser.add_argument("-b","--batchsize",default=64,type=int,help="Batch size of a training batch at each iteration")
parser.add_argument("--optim",default="sgd",help="The optimization algorithms", choices=["sgd","adam"])
parser.add_argument("--lr",default=0.001,type=float,help="Learning rate of optimization algorithms")
parser.add_argument("--epoch",default=100,type=int,help="The number of trainng episode")
parser.add_argument("--iteration",default=10000,type=int,help="The number of training iteration")
parser.add_argument("--lrscheduler",action="store_true",help="using multisteplr for training")
parser.add_argument("--lrschstep",default="50",nargs="+",type=int,help="using multisteplr for training")

# Redis related
parser.add_argument("--edgenum",default=1,type=int,help="The number of edge")
parser.add_argument("--host",default="localhost",help="The ip of Redis server")
parser.add_argument("--port",default=6379,type=int,help="The port which Redis server listen to")

# Non-critical remove related
parser.add_argument("--noncriticalremove",action="store_true",help="if remove non-critical training samples")
parser.add_argument("--strategy",default="fixed",help="The strategy of identify non-critical samples",
choices=["fixed","mean","sampler"])
parser.add_argument("--fixedratio",default=0.5,type=float,help="The ratio of selected critical samples")

# tensorboard
parser.add_argument("--tensorboard",action="store_true",help="if user tensorboard to show training process")
parser.add_argument("--summary",default=".",help="The path of summary")

# GPU
parser.add_argument("--gpu",action="store_true",help="if use gpu for training")
parser.add_argument("--gpuindex",default=0,type=int,help="the index of gpu used for training")


def validation(model,testloader,device,topk=(1,5)):
    if not isinstance(topk,tuple) and isinstance(topk,int):
        topk = (topk,)
    correct = [0 for _ in range(len(topk))]
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.squeeze().to(device)
            total += labels.size(0)

            outputs = model(images).squeeze()

            y_labels = labels.view(-1,1)
            for i in range(len(topk)):
                _, pred = outputs.topk(topk[i],1,True,True)
                correct[i] += torch.eq(pred,y_labels).sum().float().item()
    accuracy = list(map(lambda v: v*100.0/total, correct))
    return total, list(zip(topk, accuracy))


def critical_identify(model,aggdataset,criterion_loss,device,args):
    train_data = aggdataset.train_data
    train_labels = aggdataset.train_labels
    # data: n x zip x features
    # labels: n x zip
    if not args.noncriticalremove:
        ndata = train_data[:,1:,:]
        ndata = torch.reshape(ndata,args.shape)
        nlabels = train_labels[:,1:]
        nlabels = nlabels.reshape(-1,1)

        return dataset.MPRData(datax=ndata, targetsx=nlabels, shape=args.shape)
    else:
        cri_data = train_data[:,0,:] # n x feature
        cri_data = torch.reshape(cri_data, args.shape)
        cri_label = train_labels[:,0] # n
        cri_label = cri_label.reshape(-1,1)

        # calculate the loss of each aggregated points
        loss_info = []
        with torch.no_grad():
            offsetl = 0
            while offsetl < len(cri_data):
                endl = offsetl + args.batchsize
                images = cri_data[offsetl:endl]
                labels = cri_label[offsetl:endl].squeeze()

                # move data and label to `device`
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images).squeeze()
                loss = criterion_loss(outputs, labels)

                for ix in range(len(loss)):
                    loss_info.append((offsetl+ix,loss[ix]))

                offsetl += args.batchsize # move to next batch
        # sort loss info from large to small according loss
        loss_info = sorted(loss_info,key=lambda x: x[1],reverse=True)
        sel_index = []
        offset = int(args.fixedratio * len(loss_info) + 0.5 )
        for i in range(0,offset):
            sel_index.append(loss_info[i][0])
        sel_index = sorted(sel_index)
        curtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print("[{}] {} critical agg points are selected".format(curtime,len(sel_index)))

        ndata = train_data[sel_index][:,1:,:]
        ndata = torch.reshape(ndata, args.shape)
        nlabels = train_labels[sel_index][:,1:]
        nlabels = nlabels.reshape(-1,1)
        return dataset.MPRData(datax=ndata,targetsx=nlabels,shape=args.shape)




def training(args,*k,**kw):
    # if use gpus
    device = torch.device("cuda:{}".format(args.gpuindex) if torch.cuda.is_available() and args.gpu else "cpu")
    print("user device: {}".format(device))

    # redis helper related
    redis_helper = redishelper.GoSGDHelper(host=args.host, port=args.port)
    redis_helper.signin()
    while redis_helper.cur_edge_num() < args.edgenum:
        time.sleep(1) # sleep 1 second

    model_score = 1.0 / args.edgenum # the initial model parameters score

    # log_file and summary path

    log_file = "{0}-{1}-edge-{2}.log".format(time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time())),
    args.model,redis_helper.ID)
    log_dir = "tbruns/{0}-{1}-cifar10-edge-{2}".format(time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())),args.model,redis_helper.ID)

    logger = open(log_file,'w')
    swriter = SummaryWriter(log_dir)

    # load traing data
    trainset = dataset.AGGData(root=args.dataset, train=True, download=False, transform=None)

    testset = dataset.AGGData(root=args.dataset, train=False, download=False, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # construct neural network
    net = None
    if args.model == "lenet5":
        net = models.LeNet5()
    elif args.model == "resnet18":
        net = models.ResNet18()
    elif args.model == "alexnet":
        net = models.AlexNet(args.num_classes)
    elif args.model == "alexnetimg8":
        net = models.AlexNetImg8(args.num_classes)
    elif args.model == "squeezenet":
        net = models.SqueezeNet()
    elif args.model == "mobilenetv2":
        net = models.MobileNetV2()
    elif args.model == "resnet34":
        net = models.ResNet34()
    elif args.model == "resnet50":
        net = models.ResNet50()
    elif args.model == "resnet101":
        net = models.ResNet101()
    else:
        net = models.ResNet152()
    net.to(device)

    # define optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=list(args.lrschstep), gamma=0.1)

    # start training
    wallclock = 0.0
    iteration = 0 # global iterations
    for epoch in range(0,args.epoch,1):
        starteg = time.time()
        # merge parameters of other edge
        if epoch > 0:
            mintime,maxtime,param_list = redis_helper.min2max_time_params()
            print("The min/max time cost of last epoch: {}/{}".format(mintime,maxtime))
            for item in param_list:
                w1 = model_score / (model_score + item[0])
                w2 = item[0] / (model_score + item[0])

                for local,other in zip(net.parameters(),item[1]):
                    local.data = local.data * w1 + other.data.to(device) * w2
                model_score = model_score + item[0]

            while redis_helper.finish_update() is False:
                time.sleep(1.0)

        critical_extra_start = time.time()
        # identify critical training samples
        critrainset = critical_identify(net,trainset,criterion_loss,device,args)
        critrainloader = torch.utils.data.DataLoader(critrainset, batch_size=args.batchsize, shuffle=True, num_workers=0)

        critical_extra_cost = time.time() - critical_extra_start
        training_start = time.time()

        running_loss = 0.0
        record_running_loss = 0.0
        for i, data in enumerate(critrainloader, 0):
            iteration += 1
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            record_running_loss += loss.item()
            if i % 10 == 9:
                swriter.add_scalar("Training loss",record_running_loss / 10,epoch*len(critrainloader)+i)
                record_running_loss = 0.0

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        training_cost = time.time() - training_start

        # push time and parameters to Redis
        model_score = model_score / 2
        sel_edge_id = redis_helper.random_edge_id(can_be_self=True)
        paramls = list(map(lambda x: x.cpu(),list(net.parameters())))
        redis_helper.ins_time_params(sel_edge_id,training_cost,model_score,paramls)
        while not redis_helper.finish_push():
            time.sleep(1.0)

        wallclock += time.time() - starteg

        total, kaccuracy = validation(net,testloader,device,topk=(1,5))

        curtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        _header="[ {} Epoch {} /Iteration {} Wallclock {}]".format(curtime,epoch+1,iteration, wallclock)

        print('{} Accuracy of the network on the {} test images: {} %'.format(_header, total, kaccuracy_str(kaccuracy)))
        logger.write('{},{},{},{}\n'.format(epoch+1 ,iteration, wallclock, accuracy_str(kaccuracy)))
        logger.flush() # write to disk

        for item in kaccuracy:
            swriter.add_scalar("Top{}Accuracy".format(item[0]), item[1], epoch)

        # adopt learning rate of optimizer
        if args.lrscheduler:
            lr_scheduler.step()

    print('Finished Training')

    redis_helper.register_out()
    logger.close() # close log file writer

    return net

def accuracy_str(kaccuracy):
    accuracy = ""
    for i in range(len(kaccuracy)):
        item = kaccuracy[i]
        if i == 0:
            accuracy = accuracy + "{}".format(item[1])
        else:
            accuracy = accuracy + ",{}".format(item[1])
    return accuracy


def kaccuracy_str(kaccuracy):
    accuracy = ""
    for i in range(len(kaccuracy)):
        item = kaccuracy[i]
        if i == 0:
            accuracy = accuracy + "Top{}: {} %".format(item[0],item[1])
        else:
            accuracy = accuracy + " Top{}: {} %".format(item[0],item[1])
    return accuracy

if __name__ == '__main__':
    args = parser.parse_args()
    training(args)
