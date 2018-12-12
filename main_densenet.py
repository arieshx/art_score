# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import os
import csv
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from data_loader import AVADataset
from PIL import Image
# from model import *
from densenet import *
from loss import emd_loss

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, transform=train_transform)
    valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
                                             shuffle=False, num_workers=config.num_workers)

    ###加载模型
    densenet = models.densenet121(pretrained=True)
    class_in = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
            # nn.Dropout(p=0.75),
            nn.Linear(in_features=class_in, out_features=10),
            nn.Softmax())
    model = densenet
#     ###读取模型参数
#     pretrained_dict = densenet.state_dict()
#     model_dict = model.state_dict()
#     ## 将pretrained_dict里不属于model_dict的键剔除掉
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in densenet.classifier.state_dict().items()}
#     ## 更新现有的model_dict
#     model_dict.update(pretrained_dict)
#     ## 加载我们真正需要的state_dict
#     model.load_state_dict(model_dict)
#     # model = NIMA(base_model)
    
    #
   
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.classifier.parameters(), 'lr': dense_lr}],lr = conv_base_lr,
        momentum=0.9)

    opt_Adam  = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

    # send hyperparams
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                outputs = model(images)
                outputs = outputs.view(-1, 10, 1)

                # print(outputs)
                optimizer.zero_grad()
                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()

                # lrs.send('train_emd_loss', loss.item())

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (
                epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d averaged training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if (epoch + 1) % 5 == 0:
                conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9)

            # do validation after each epoch
            batch_val_losses = []
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)

            print('Epoch %d completed. Averaged EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.mkdir(config.ckpt_path)
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pkl' % (epoch + 1)))
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print(
                        'Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)

            with open('./output/loss/emd_densenet.txt', 'w') as f:
                f.write(' '.join(epochs)+'\n')
                f.write(' '.join(train_losses)+'\n')
                f.write(' '.join(val_losses))

            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./output/loss/emd_densenet.png')

    DICT_Pred = dict()
    count = 0
    image_num = 0
    total_num = 0
    if config.test:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-34.pkl')))
        # model.load_state_dict(torch.load('epoch-5.pkl'))
        model.eval()
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.test_img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False)

        for data in test_loader:
            total_num = len(test_loader)
            img_path = data['img_id']
            image = data['image'].to(device)
            score = data['score'].to(device).float()
            output = model(image)
            output = output.view(10, 1)

            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            predicted_mean = predicted_mean.cpu().detach().numpy()
            images_ = Image.open(img_path[0])

            #title = str(int(np.rint(predicted_mean[0]*10+20)))
            title = str(int(np.rint(predicted_mean[0]*12)))
            compare = 'machine:'+title  +' , '+'human:'+str(int(score.cpu().detach().numpy()[0]))
            plt.title(compare )
            fname = img_path[0].split('/')[-1].split('.')[0]
            if title in DICT_Pred:
                DICT_Pred[title].append(fname)
            else:
                DICT_Pred[title] = [fname]
            #plt.imshow(images_)
            #plt.savefig('img/'+ img_path[0].split('/')[-1].split('.')[0]+'.png')
            if (abs(int(title) - int(score.cpu().detach().numpy()[0])) <= 5):
                count = count + 1
            # plt.show()

    predition = count / total_num
    for fname_label in DICT_Pred:
        LIST_label_train = DICT_Pred[fname_label]
        print(fname_label, len(LIST_label_train))
        with open('prediction_10_0.csv', 'ab') as output_file:
            LIST_ = []
            LIST_.append(int(fname_label))
            LIST_.append(len(LIST_label_train))
            csv_writer = csv.writer(output_file, dialect='excel')
            csv_writer.writerow(LIST_)
    with open('prediction_10_0.csv', 'ab') as output_file:
        LIST_ = []
        LIST_.append('prediction')
        LIST_.append(predition)
        csv_writer = csv.writer(output_file, dialect='excel')
        csv_writer.writerow(LIST_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/media/haoxin/A/data_work/art_rank/resize_img')
    parser.add_argument('--val_img_path', type=str, default='/media/haoxin/A/data_work/art_rank/resize_img')
    parser.add_argument('--test_img_path', type=str, default='/media/haoxin/A/data_work/art_rank/resize_img')
    parser.add_argument('--train_csv_file', type=str, default='/media/haoxin/A/data_work/art_rank/exam01_jh_all_train.csv')
    parser.add_argument('--val_csv_file', type=str, default='/media/haoxin/A/data_work/art_rank/exam01_jh_all_test.csv')
    parser.add_argument('--test_csv_file', type=str, default='/media/haoxin/A/data_work/art_rank/exam01_jh_all_test.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--conv_base_lr', type=float, default=2e-7)
    parser.add_argument('--dense_lr', type=float, default=2e-7)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./output/model')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--warm_start_epoch', type=int, default=12)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=True)
    config = parser.parse_args()
    main(config)

