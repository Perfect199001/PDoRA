import argparse
import os
import random
import logging
import numpy as np
import time

import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.optim

#from model.fine_tuning.ft_swinunetr.swin_unetr_pissa1 import SwinUNETR
from model.fine_tuning.ft_swinunetr.swin_unetr_all_ft import SwinUNETR
import torch.distributed as dist

from model.pretrain import criterions
from model.pretrain.criterions import*
from data_process.BraTS0 import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn
#from models.SwinTransformer import *
#from models.SwinUNETR import SwinUNETR
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']  = '1'
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'
#os.environ["RANK"] = "0"
#os.environ['WORLD_SIZE'] = '1'

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default='hxyang', type=str)
parser.add_argument('--experiment', default='ft_swinunetr', type=str)
#parser.add_argument('--date', default='2023-12-07', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description',
                    default='SwinUNETR,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/media/ubuntu2204/Elements/kits23/dataset/', type=str)
parser.add_argument('--train_dir', default='/media/ubuntu2204/Elements/kits23/dataset/', type=str)
parser.add_argument('--val_dir', default='/media/ubuntu2204/Elements/kits23/dataset/', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)
parser.add_argument('--val_file', default='val.txt', type=str)
parser.add_argument('--dataset', default='kits23', type=str)
parser.add_argument('--model_name', default='multitask', type=str)

# Training Information
parser.add_argument('--lr', default=2e-3, type=float)
parser.add_argument('--weight_decay', default=6e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='CE_DICE', type=str)
parser.add_argument('--seg_num', default=2, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--seed', default=500, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--num_workers', default=2, type=int)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=1000, type=int)
parser.add_argument('--val_epoch', default=10, type=int)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


class Finetune_UNETR:
    def __init__(self, resume_path, in_channels, out_channels, img_size, r):
        self.resume_path = resume_path
        self.r = r
        self.model = SwinUNETR(img_size=64, in_channels=1, out_channels=2)
        self.load_model()
        self.W2_parameters = []
        self.column_norms_before_finetune = []

    def load_model(self):
        checkpoint = torch.load(self.resume_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def compute_new_weights(self):
        weights = []
        for name, param in self.model.named_parameters():
            if "linear1.weight" in name and len(param.shape) == 2 and param.numel() == 3072 * 768:
                reshaped_param = param.data.view(4, 768, 768)
                weights.append(reshaped_param)
            elif "linear2.weight" in name and len(param.shape) == 2 and param.numel() == 768 * 3072:
                reshaped_param = param.data.view(4, 768, 768)
                weights.append(reshaped_param)

        W1_list = []
        W2_list = []
        m_list = []
        
        def process_weight(param, r):
            # SVD 分解
            U, S, V = torch.svd(param)
            S_diag = torch.diag(S)

            # 计算 W2 部分
            U2 = U[:, :r]
            S2 = S_diag[:r, :r]
            V2 = V[:, :r]
            W2 = torch.mm(U2, torch.mm(S2, V2.T))
            
            # 计算列范数 m
            m = W2.norm(dim=0)
            m_list.append(m)

            # 分解后的 W1 部分
            U1 = U[:, r:]
            S1 = S_diag[r:, r:]
            V1 = V[:, r:]
            W1 = torch.mm(U1, torch.mm(S1, V1.T))
            W1_list.append(W1)

            # 将 U2, S2, V2 转化为可训练参数
            U2 = nn.Parameter(U2.detach().clone())
            S2 = nn.Parameter(S2.detach().clone())
            V2 = nn.Parameter(V2.detach().clone())
            self.W2_parameters.extend([U2, S2, V2])

            W2_list.append(W2)

        for weight in weights:
            for i in range(weight.shape[0]):
                process_weight(weight[i], self.r)

        self.W1_list = W1_list
        self.W2_list = W2_list
        self.column_norms_before_finetune = m_list

        return W1_list, W2_list

    def update_weights(self, W1_list, W2_list):
        idx = 0
        for name, param in self.model.named_parameters():
            if "linear1.weight" in name and len(param.shape) == 2 and param.numel() == 3072 * 768:
                reshaped_param = param.data.view(4, 768, 768)
                for i in range(reshaped_param.shape[0]):
                    W2_normalized = W2_list[idx] / W2_list[idx].norm(dim=0, keepdim=True)
                    W2_adjusted = W2_normalized * self.column_norms_before_finetune[idx]
                    reshaped_param[i] = W2_adjusted + W1_list[idx]
                    idx += 1
                param.data = reshaped_param.view(param.shape)
            
            elif "linear2.weight" in name and len(param.shape) == 2 and param.numel() == 768 * 3072:
                reshaped_param = param.data.view(4, 768, 768)
                for i in range(reshaped_param.shape[0]):
                    W2_normalized = W2_list[idx] / W2_list[idx].norm(dim=0, keepdim=True)
                    W2_adjusted = W2_normalized * self.column_norms_before_finetune[idx]
                    reshaped_param[i] = W2_adjusted + W1_list[idx]
                    idx += 1
                param.data = reshaped_param.view(param.shape)

    def freeze_parameters(self):
        for name, param in self.model.named_parameters():
            if not any(key in name for key in ["decoder", "mlp.linear1", "mlp.linear2"]):
                param.requires_grad = False
                
        for param in self.W2_parameters:
            param.requires_grad = True
            
        # 确保 m 的参数是可训练的
        for param in self.m_parameters:
            param.requires_grad = True

    def finetune(self):
        W1_list, W2_list = self.compute_new_weights()
        if W1_list and W2_list:
            self.update_weights(W1_list, W2_list)
            self.freeze_parameters()


    
def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log_pissa_dora','qkv_L','kits','train','bs=1','r=16')
    log_file = log_dir + '.txt' 
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Example usage
    finetuner = Finetune_UNETR(
        resume_path='/home/ubuntu2204/yhx/SegMamba-main/checkpoint_pre_CE/pre_swinunetr2024-06-04/model_epoch_last.pth',
        in_channels=1,
        out_channels=2,
        img_size=64,
        r=16,
    )
    
    finetuner.finetune()
    model = finetuner.model
    #model = SwinUNETR(img_size=128, in_channels=1, out_channels=2, feature_size=48)
    #model = SegMamba(in_chans=1,out_chans=2)
    
    

    model.cuda()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    # 检查哪些参数将会被优化
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter to be optimized: {name}")
     # 计算模型的总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数量: {total_params}")    
    # 计算可训练参数的总量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练的参数量: {trainable_params}")        
            
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)


    criterion = getattr(criterions, args.criterion)

 
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint_pissa_dora','qkv_L','kits','train','bs=1','r=16')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    #resume = './checkpoint/SwinUNETR2023-11-04/model_epoch_399.pth'#/home/ubuntu2204/cwg/UVSnet/checkpoint/UNETR2024-05-23/model_epoch_last.pth

    writer = SummaryWriter()



    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.val_dir, args.val_file)
    val_root = os.path.join(args.root, args.val_dir)

    train_set = BraTS(train_list, train_root, args.mode)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_set = BraTS(val_list, val_root, mode = 'vaild')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    #记录数据集中的样本数train_set
    logging.info('Samples for train = {}'.format(len(train_set)))
    #记录数据集中的样本数val_set
    logging.info('Samples for val = {}'.format(len(val_set)))



    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch): 
        torch.manual_seed(epoch)
        train_loader.dataset.shuffle()
        
        #设置当前进程的标题包括用户名，当前epoch，以及epoch总数
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        #记录当前epoch的开始时间
        #start_epoch = time.time()

        #train

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            #print((target==1).sum())

            output = model(x)

            #计算模型预测与真实输出相比的损失，并返回多个损失值
            loss,loss_0,loss_1 = softmax_dice(output, target) 
            reduce_loss = loss.data.cpu().numpy()
            reduce_loss0 = loss_0.data.cpu().numpy()
            reduce_loss1 = loss_1.data.cpu().numpy()
            
            if args.local_rank == 0:
                #打印有关当前时期和迭代的一些信息，以及损失及其组成部分。
                logging.info('Epoch: {}_Iter:{}  Dice_loss: {:.5f}|0:{:.4f} |1:{:.4f} |'
                           .format(epoch, i, reduce_loss,reduce_loss0, reduce_loss1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        end_epoch = time.time()

        #val
        if epoch%args.val_epoch==0:
             logging.info('Samples for val = {}'.format(len(val_set)))
             with torch.no_grad():
                 for i, data in enumerate(val_loader):
                     #adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
                     x, target = data
                     x = x.cuda(args.local_rank, non_blocking=True)
                     #将目标移动到与输入相同的 GPU。
                     target = target.cuda(args.local_rank, non_blocking=True)
                     
                     output = model(x)
                     
                     lossSEG, loss_0,loss_1 = softmax_dice(output,target)
                     
                     if args.local_rank == 0:
                          
                         logging.info('Epoch: {}_Iter:{}  Dice:SEG: {:.4f} |Dice_0:{:.4f}| Dice_1:{:.4f}|'
                             .format(epoch, i,  lossSEG, loss_0, loss_1))
             
             
        end_epoch = time.time()  
        
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)



            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss:', reduce_loss, epoch)
            writer.add_scalar('loss0:', reduce_loss0, epoch)
            writer.add_scalar('loss1:', reduce_loss1, epoch)
   
        if args.local_rank == 0:
            #计算并记录当前epoch时间消耗和估计的剩余训练时间
            epoch_time_minute = (end_epoch-args.start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60

            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

        

    if args.local_rank == 0:
        #writer.close()函数关闭 TensorBoard writer 对象，该对象将累积的摘要数据写入摘要文件。
        writer.close()
        
        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)

    #end_time记录训练过程结束的时间，total_time变量计算总训练时间        
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    #logging.info()函数将计算值作为信息性消息记录到日志输出中
    #第一条消息记录总训练时间，第二条消息表示训练过程已完成
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')





def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()



