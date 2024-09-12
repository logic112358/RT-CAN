import os
import argparse
import time
import datetime
import stat
import shutil
import random
import warnings
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop
from util.util import compute_results, visualize, compute_results2
from sklearn.metrics import confusion_matrix
from loss_hub.loss_join import JointLoss, MRLoss
from loss_hub import DiceLoss, SoftCrossEntropyLoss, FocalLoss, FusionLoss
from torch.cuda.amp import autocast, GradScaler
# from model.AGSNet import GasSegNet
from model.RT_CAN import GasSegNet
import sys

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str,
                    default='ResNet152')
parser.add_argument('--batch_size', '-b', type=int, default=4)
parser.add_argument('--seed', default=3409, type=int,
                    help='seed for initializing training.')
parser.add_argument('--lr_start', '-ls', type=float, default=0.02)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.97)
parser.add_argument('--epoch_max', '-em', type=int, default=50)
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='F:\\WJ\\GasSeg\\dataset')
parser.add_argument('--weight', '-f', type=str, default=None)
args = parser.parse_args()
##############################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda")

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
]

scaler = GradScaler()



def train(epo, model, train_loader, optimizer):
    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        with autocast():
            start_t = time.time()  # time.time() returns the current time
            optimizer.zero_grad()
            DiceLoss_fn = DiceLoss(mode='multiclass')
            SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
            criterion = JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                                  first_weight=0.5, second_weight=0.5).cuda()
            # modality_reduce_loss = MRLoss().cuda()
            logits_S, logits_T, fuse, thermal, rgb = model(images)
            loss_1 = criterion(logits_S, labels)
            loss_2 = criterion(logits_T, labels)
            loss = loss_1 + loss_2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        if it % 2 == 0:
            print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s'
                  % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                     len(names) / (time.time() - start_t), float(loss),
                     datetime.datetime.now().replace(microsecond=0) - start_datetime))
            print(
                f'logits_S loss: {loss_1}, logits_T loss: {loss_2}')
        if accIter['train'] % 1 == 0:
            pass
        view_figure = True  # note that I have not colorized the GT and predictions here
        if accIter['train'] % 500 == 0:
            if view_figure:
                # can only display 3-channel images, so images[:,:3]
                input_rgb_images = vutils.make_grid(
                    images[:, :3], nrow=8, padding=10)
                # label (0,1,2..) is invisable, multiply a constant for visualization
                scale = max(1, 255 // args.n_class)
                # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = labels.unsqueeze(1) * scale
                # change to 3-channel for visualization
                groundtruth_tensor = torch.cat(
                    (groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)
                groudtruth_images = vutils.make_grid(
                    groundtruth_tensor, nrow=8, padding=10)
                # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = logits_T.argmax(1).unsqueeze(1) * scale
                # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_tensor = torch.cat(
                    (predicted_tensor, predicted_tensor, predicted_tensor), 1)
                predicted_images = vutils.make_grid(
                    predicted_tensor, nrow=8, padding=10)
        accIter['train'] = accIter['train'] + 1

def testing2(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "gas"]
    testing_results_file = os.path.join(weight_dir, 'testing_results.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            # BBS使用双输出
            logit, logits,_,_,_ = model(images)
            logit_mix = (logit + logits)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logit_mix.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[
                                    0, 1])
            conf_total += conf
    precision, recall, IoU, F1, F2, _ = compute_results2(conf_total)

    if epo == 0:
        with open(testing_results_file, 'w') as f:
            # f.write(
            #     "# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)\n")
            f.write(
                "# epoch: unlabeled, gas, average(nan_to_num), (precision %,(recall)Acc %, IoU %, F1 ,F2)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f, %0.4f|' % (100*precision[i], 100 * recall[i], 100 * IoU[i], 100* F1[i], 100* F2[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f, %0.4f| \n' % (
            100 * np.mean(np.nan_to_num(precision)), 100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU)), 100 * np.mean(np.nan_to_num(F1)), 100 * np.mean(np.nan_to_num(F2))))
    print('saving testing results.')

if __name__ == '__main__':
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    if args.model_name == "ResNet50":
        num_resnet_layers = 50
    elif args.model_name == "ResNet152":
        num_resnet_layers = 152
    else:
        sys.exit('no such type model.')

    model = GasSegNet(args.n_class,num_resnet_layers=num_resnet_layers)
    # model = EAEFNet(args.n_class)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    if args.weight:
        model_dir = os.path.join('./runs',args.model_name)
        weight = os.path.join(args.weight)
        print('loading model file %s... ' % args.weight)
        pretrained_weight = torch.load(
            weight, map_location=lambda storage, loc: storage.cuda(args.gpu))
        own_state = model.state_dict()
        for name, param in pretrained_weight.items():
            if name not in own_state:
                print(name)
                continue
            own_state[name].copy_(param)
        print('done!')
        for name, param in pretrained_weight.items():
            if name not in own_state:
                print(name)
                continue
            own_state[name].copy_(param)
        print('done!')

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.8, patience=4)


    weight_dir = os.path.join("./checkpoints/", args.model_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    os.chmod(weight_dir,
             stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(
        data_dir=args.data_dir, 
        split='train', 
        transform=augmentation_methods,
        input_w=640,
        input_h=512
        )

    test_dataset = MF_dataset(
        data_dir=args.data_dir, 
        split='test',
        input_w=640,
        input_h=512
        )

    val_dataset = test_dataset


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        train(epo, model, train_loader, optimizer)
        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '_latest.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(model.state_dict(), checkpoint_model_file)
        testing2(epo, model, test_loader)
        scheduler.step()
