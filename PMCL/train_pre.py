'''
@contact:xind2023@mail.ustc.edu.cn
@time:2025/9/1
'''

import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import time
import torch.nn.functional as F
import torch.nn as nn
import math

from utils.dataset.isic import isic2018_dataset, isic2019_dataset, augmentation_rand, augmentation_sim,augmentation_test
from utils.eval_metrics import ConfusionMatrix, Auc
from models.pmcl import PMCL_model,balanced_proxies
from models.CBCL_loss import CBCLLoss
from models.pre_CBAM_loss import CBAMLoss
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

n=8 #the number of classes

def initialize_column_vector(n):
    global global_column_vector
    global_column_vector = torch.zeros((n), dtype=torch.float).cuda()
    return global_column_vector

accuracy_per_class_global = initialize_column_vector(n)

'''function for saving model'''
def model_snapshot(model,new_modelpath,old_modelpath=None,only_bestmodel=False):
    if only_bestmodel and old_modelpath:
        os.remove(old_modelpath)
    torch.save(model.state_dict(),new_modelpath)

'''function for getting proxies number'''
def get_proxies_num(cls_num_list):
    m_list = 1.0 / np.sqrt(cls_num_list)
    max_m = (max(cls_num_list) / min(cls_num_list))//10+2
    m_list = (m_list * max_m) / max(m_list)
    m_list = m_list
    m_list = np.ceil(m_list).astype(np.int)
    print(m_list)
    return m_list

def main(args):
    global accuracy_per_class_global

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    log_file = open(os.path.join(args.log_path,'train_log.txt'), 'w')

    '''print args'''
    for arg in vars(args):
        print(arg, getattr(args, arg))
        print(arg, getattr(args, arg),file=log_file)


    '''load models'''
    model = PMCL_model(num_classes=args.num_classes,feat_dim=args.feat_dim)
    proxy_num_list = get_proxies_num(args.cls_num_list)
    model_proxy = balanced_proxies(dim=args.feat_dim,proxy_num=sum(proxy_num_list))

    if args.cuda:
        model.cuda()
        model_proxy.cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0),file=log_file)
    print("Model_proxy size: {:.5f}M".format(sum(p.numel() for p in model_proxy.parameters())/1000000.0))
    print("Model_proxy size: {:.5f}M".format(sum(p.numel() for p in model_proxy.parameters())/1000000.0),file=log_file)
    print("=============model init done=============")
    print("=============model init done=============",file=log_file)

    '''load optimizer'''
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_proxies = optim.SGD(model_proxy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  momentum=0.9)

    # cosine lr
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lr_scheduler_proxies = optim.lr_scheduler.CosineAnnealingLR(optimizer_proxies, T_max=args.epochs)

    complete = False

    '''load dataset'''
    transfrom_train = [augmentation_rand, augmentation_sim]
    if args.dataset == 'ISIC2018':
        train_iterator = DataLoader(isic2018_dataset(path=args.data_path, transform=transfrom_train, mode='train'),
                                    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_iterator = DataLoader(isic2018_dataset(path=args.data_path, transform=augmentation_test, mode='valid'),
                                    batch_size=1, shuffle=False, num_workers=2)
        test_iterator = DataLoader(isic2018_dataset(path=args.data_path, transform=augmentation_test, mode='test'),
                                   batch_size=1, shuffle=False, num_workers=2)

    elif args.dataset == 'ISIC2019':
        train_iterator = DataLoader(isic2019_dataset(path=args.data_path, transform=transfrom_train, mode='train'),
                                    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_iterator = DataLoader(isic2019_dataset(path=args.data_path, transform=augmentation_test, mode='valid'),
                                    batch_size=32, shuffle=False, num_workers=2)
        test_iterator = DataLoader(isic2019_dataset(path=args.data_path, transform=augmentation_test, mode='test'),
                                      batch_size=32, shuffle=False, num_workers=2)
    else:
        raise ValueError("dataset error")

    '''load loss'''
    a=1
    b=2
    if args.ce == 'CBAM':
        criterion_ce = CBAMLoss(cls_num_list=args.cls_num_list, class_difficulty = 1 - accuracy_per_class_global, alpha = a, p = b, beta = (1 - 1/(1+math.exp(-1)))/30, E1 = args.E1, E2 = args.E2).cuda(args.gpu)
    if args.ce == 'CrossEntropy':
        criterion_ce = torch.nn.CrossEntropyLoss()

    if args.con == 'CBCL':
        criterion_cbcl = CBCLLoss(cls_num_list=args.cls_num_list, proxy_num_list=proxy_num_list, class_difficulty = 1 - accuracy_per_class_global, alpha = a, p = b, beta = (1 - 1/(1+math.exp(-1)))/30)
    
    alpha = args.alpha
    beta = args.beta

    '''train'''
    best_acc = 0.0
    old_model_path = None
    curr_patience = args.patience
    start_time = time.time()
    try:
        for e in range(args.epochs):
            model.train()
            model_proxy.train()
            print('Epoch:{}'.format(e))
            print('Epoch:{}'.format(e),file=log_file)

            start_time_epoch = time.time()
            train_loss = 0.0

            lr_scheduler.step()
            lr_scheduler_proxies.step()
            optimizer_proxies.zero_grad()

            for batch_index, (data, label) in enumerate(train_iterator):

                if args.cuda:
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                    label = label.cuda()
                diagnosis_label = label.squeeze(1)

                optimizer.zero_grad()

                output, feat_mlp = model(data)
                output_proxy = model_proxy()

                feat_mlp = torch.cat([feat_mlp[0].unsqueeze(1), feat_mlp[1].unsqueeze(1)], dim=1)

                if args.ce == 'CBAM':
                    loss_ce = criterion_ce(output, diagnosis_label, (e+1), 1 - accuracy_per_class_global)
                if args.ce == 'CrossEntropy':
                    loss_ce = criterion_ce(output, diagnosis_label)

                
                if args.con == 'CBCL':
                    loss_bhp = criterion_cbcl(output_proxy, feat_mlp, diagnosis_label, 1 - accuracy_per_class_global)
                
                loss = alpha * loss_ce + beta * loss_bhp
                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                if batch_index % 50 == 0 and batch_index != 0:
                    predicted_results = torch.argmax(output, dim=1)
                    correct_num = (predicted_results.cpu() == diagnosis_label.cpu()).sum().item()
                    acc = correct_num / len(diagnosis_label)
                    print('Training epoch: {} [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Learning rate: {}'.format(e,
                        batch_index * args.batch_size, len(train_iterator.dataset), loss.item(), acc, optimizer.param_groups[0]['lr']))
                    print('Training epoch: {} [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Learning rate: {}'.format(e,
                        batch_index * args.batch_size, len(train_iterator.dataset), loss.item(), acc, optimizer.param_groups[0]['lr']),file=log_file)


            optimizer_proxies.step()

            print("Epoch {} complete! Average Training loss: {:.4f}".format(e, train_loss / len(train_iterator)))
            print("Epoch {} complete! Average Training loss: {:.4f}".format(e, train_loss / len(train_iterator)),file=log_file)


            '''validation'''
            model.eval()
            model_proxy.eval()
            pro_diag, lab_diag = [], []
            val_confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
            with torch.no_grad():
                for batch_index, (data, label) in enumerate(valid_iterator):
                    if args.cuda:
                        data = data.cuda()
                        label = label.cuda()
                    diagnosis_label = label.squeeze(1)

                    output = model(data)
                    predicted_results = torch.argmax(output, dim=1)
                    pro_diag.extend(output.detach().cpu().numpy())
                    lab_diag.extend(diagnosis_label.cpu().numpy())

                    val_confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())

                dia_acc, accuracy_per_class_global = val_confusion_diag.summary(log_file)
                print(1-accuracy_per_class_global)
                Auc(pro_diag, lab_diag, args.num_classes, log_file)
                f_score_list = val_confusion_diag.get_f1score()

                end_time_epoch = time.time()
                training_time_epoch = end_time_epoch - start_time_epoch
                total_training_time = time.time() - start_time
                remaining_time = training_time_epoch * args.epochs - total_training_time
                print("Total training time: {:.4f}s, {:.4f} s/epoch, Estimated remaining time: {:.4f}s".format(
                    total_training_time, training_time_epoch, remaining_time))
                print("Total training time: {:.4f}s, {:.4f} s/epoch, Estimated remaining time: {:.4f}s".format(
                    total_training_time, training_time_epoch, remaining_time),file=log_file)

                if dia_acc > best_acc:
                    curr_patience = args.patience
                    best_acc = dia_acc
                    new_model_path = os.path.join(args.model_path, 'bestacc_model_{}.pth'.format(e))
                    model_snapshot(model, new_model_path, old_modelpath=old_model_path,only_bestmodel=True)
                    old_model_path = new_model_path
                    print("Found new best model, saving to disk...")
                else:
                    curr_patience -= 1
                    if curr_patience == 0:
                        print("Early stopping, best accuracy: {:.4f}".format(best_acc))
                        print("Early stopping, best accuracy: {:.4f}".format(best_acc),file=log_file)
                        complete = True
                        break
                if (e + 1)%10 == 0:
                    new_model_path = os.path.join(args.model_path, 'model_{}.pth'.format(e+1))
                    model_snapshot(model, new_model_path, old_modelpath=old_model_path,only_bestmodel=False)
                    old_model_path = new_model_path

                if e == args.epochs - 1:
                    print("Training complete, best accuracy: {:.4f}".format(best_acc))
                    print("Training complete, best accuracy: {:.4f}".format(best_acc),file=log_file)
                    complete = True
            log_file.flush()

        '''test'''
        if complete:

            for e in range(6, args.epochs//10):
                model.load_state_dict(torch.load(os.path.join(args.model_path, 'model_{}.pth'.format((9+1)*10))),strict=True)
                model.eval()

                pro_diag, lab_diag = [], []
                confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
                with torch.no_grad():
                    for batch_index, (data, label) in enumerate(test_iterator):
                        if args.cuda:
                            data = data.cuda()
                            label = label.cuda()
                        diagnosis_label = label.squeeze(1)

                        output = model(data)
                        predicted_results = torch.argmax(output, dim=1)
                        pro_diag.extend(output.detach().cpu().numpy())
                        lab_diag.extend(diagnosis_label.cpu().numpy())

                        confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())

                    print("Test confusion matrix:")
                    print("Test confusion matrix:",file=log_file)
                    confusion_diag.summary(log_file)
                    Auc(pro_diag, lab_diag, args.num_classes, log_file)
                    log_file.flush()

            model.load_state_dict(torch.load(old_model_path),strict=True)
            model.eval()

            pro_diag, lab_diag = [], []
            confusion_diag = ConfusionMatrix(num_classes=args.num_classes, labels=list(range(args.num_classes)))
            with torch.no_grad():
                for batch_index, (data, label) in enumerate(test_iterator):
                    if args.cuda:
                        data = data.cuda()
                        label = label.cuda()
                    diagnosis_label = label.squeeze(1)

                    output = model(data)
                    predicted_results = torch.argmax(output, dim=1)
                    pro_diag.extend(output.detach().cpu().numpy())
                    lab_diag.extend(diagnosis_label.cpu().numpy())

                    confusion_diag.update(predicted_results.cpu().numpy(), diagnosis_label.cpu().numpy())

                print("Test confusion matrix:")
                print("Test confusion matrix:",file=log_file)
                confusion_diag.summary(log_file)
                print("Test AUC:")
                print("Test AUC:",file=log_file)
                Auc(pro_diag, lab_diag, args.num_classes, log_file)

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        log_file.close()


parser = argparse.ArgumentParser(description='Training for the classification task')

#dataset
parser.add_argument('--data_path', type=str, default='./data/ISIC2019/', help='the path of the data')
parser.add_argument('--dataset', type=str, default='ISIC2019',choices=['ISIC2018','ISIC2019'], help='the name of the dataset')
parser.add_argument('--model_path', type=str, default="./results/ab_newaug_gai_ISIC2019_1_2/", help='the path of the model')
parser.add_argument('--log_path', type=str, default=None, help='the path of the log')


# training parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')


# loss weights
parser.add_argument('--alpha', type=float, default=2.0, choices=[0,0.5,1.0,2.0], help='weight of the CBAM loss')
parser.add_argument('--beta', type=float, default=1.0, choices=[0,0.5,1.0,2.0],help='weight of the CBCL loss')
# hyperparameters for ce loss
parser.add_argument('--E1', type=int, default=40, choices=[20, 30, 40, 50, 60, 70],help='hyperparameter for ce loss')
parser.add_argument('--E2', type=int, default=60, choices=[30, 40, 50, 60, 70, 80],help='hyperparameter for ce loss')

# hyperparameters for model
parser.add_argument('--feat_dim', dest='feat_dim', type=int, default=128)

parser.add_argument('--con', type=str, default='CBCL')
parser.add_argument('--ce', type=str, default='CBAM')

def _seed_torch(args):
    r"""
    Sets custom seed for torch

    Args:
        - seed : Int

    Returns:
        - None

    """
    import random
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        else:
            raise EnvironmentError("GPU device not found")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _seed_torch(args)
    if args.dataset == 'ISIC2018':
        args.cls_num_list = [327, 514, 1099, 115, 1113, 6705, 142]
        args.num_classes = 7
    elif args.dataset == 'ISIC2019':
        args.cls_num_list = [867,3323,2624,239,4522,12875,628,253]
        args.num_classes = 8
    else:
        raise Exception("Invalid dataset name!")

    if args.log_path is None:
        args.log_path = args.model_path
    main(args)
    print("Done!")