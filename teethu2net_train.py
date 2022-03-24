import argparse
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import Teeth_dataloader
from model.teeth_U2NET import Teeth_U2NET
from torch.utils.tensorboard import SummaryWriter
from loss import TeethNetLoss
from torch.optim import lr_scheduler


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='TeethNet', help='model name')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_size', default=256, type=int, help='image size')

    # dataset

    parser.add_argument('--data_folder_name', default='train_data', help='data_folder_name')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--num_workers', default=4, type=int)
    config = parser.parse_args()
    return config


class Model_Train():
    def __init__(self , config):
        self.model_loss = TeethNetLoss()
        self.model_name = config['name']
        self.epoch_num = config['epochs']
        self.save_frq = 2000  # save the model every 2000 iterations
        self.batch_size_train = config['batch_size']
        self.epoch = 0
        self.root_dir = ""
        self.log_dir = os.path.join(self.root_dir, "logs/my_board/" + os.sep)
        self.pthFile_name = "NONE"
        self.writer = SummaryWriter(self.log_dir)

        # ------- 2. set the directory of training dataset --------
        data_dir = os.path.join(self.root_dir, config['data_folder_name'] + os.sep)
        tra_image_dir = os.path.join("images" + os.sep)
        tra_label_dir = os.path.join("labels" + os.sep)
        image_ext = ".jpg"
        label_ext = ".bmp"
        self.model_dir = os.path.join(self.root_dir, config['data_folder_name'], self.model_name + os.sep)
        os.makedirs(self.model_dir, exist_ok=True)
        tra_img_name_list = glob.glob(data_dir + tra_image_dir + "*" + image_ext)
        tra_lbl_name_list = []
        for img_path in tra_img_name_list:
            img_name = img_path.split(os.sep)[-1]
            aaa = img_name.split(".")
            bbb = aaa[0:-1]
            imidx = bbb[0]
            for i in range(1, len(bbb)):
                imidx = imidx + "." + bbb[i]
            tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

        print("---")
        print("train images: ", len(tra_img_name_list))
        print("train labels: ", len(tra_lbl_name_list))
        print("---")
        self.train_num = len(tra_img_name_list)
        teeth_dataloader = Teeth_dataloader(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                       transform=transforms.Compose([RescaleT(config['input_size']), ToTensorLab(flag=0)]), )
        self.teeth_dataloader = DataLoader(teeth_dataloader, batch_size=self.batch_size_train, shuffle=True , num_workers=config['num_workers'])
        self.model = Teeth_U2NET(config['input_channels'], config['num_classes'])

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(params, lr=config['lr'], betas=(0.9, 0.999), eps=1e-08,weight_decay=0, ) #weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'],weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        if config['scheduler'] == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        elif config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=config['factor'], patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=[int(e) for e in config['milestones'].split(',')],
                                                 gamma=config['gamma'])
        elif config['scheduler'] == 'ConstantLR':
            self.scheduler = None
        else:
            raise NotImplementedError

    def model_train(self):
        if self.pthFile_name != "NONE":
            checkpoint = torch.load(self.pthFile_name)
            if checkpoint["epoch"] > 0:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                epoch = checkpoint["epoch"]
                print(f"=====> epoch:{epoch}")
                if torch.cuda.is_available():
                    self.model.cuda()
                # ------- 4. define optimizer --------
                print("---define optimizer...")
                # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,weight_decay=0, )
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                if torch.cuda.is_available():
                    self.model.cuda()
                # ------- 4. define optimizer --------
                print("---define optimizer...")
                #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,weight_decay=0, )
        else:
            if torch.cuda.is_available():
                self.model.cuda()
            # ------- 4. define optimizer --------
            print("---define optimizer...")
            #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, )

        # ------- 5. training process --------
        print("---start training...")
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

        while self.epoch < self.epoch_num:
            self.model.train()
            for i, data in enumerate(self.salobj_dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1
                inputs, labels = data["image"], data["label"]
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = (
                    Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False),)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                # y zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
                loss2, loss = self.model_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)
                loss.backward()
                self.optimizer.step()
                # # print statistics
                running_loss += loss.data
                running_tar_loss += loss2.data
                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                    self.epoch + 1, self.epoch_num, (i + 1) * self.batch_size_train, self.train_num, ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,))

                self.writer.add_scalar("train loss", running_loss / ite_num4val, self.epoch + 1)
                self.writer.add_scalar("tar loss", running_tar_loss / ite_num4val)

                if ite_num % self.save_frq == 0:
                    torch.save({"epoch": self.epoch, "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(), },
                               self.model_dir + self.model_name + "_%d_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                               self.epoch, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val,), )
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    self.model.train()  # resume train
                    ite_num4val = 0

            self.epoch = self.epoch + 1
            self.scheduler.step()

        torch.save(
            {"epoch": self.epoch, "model_state_dict": self.model.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(), },
            self.model_dir + self.model_name + "_%d_final_bce_itr_%d_train_%3f_tar_%3f.pth" % (
            self.epoch, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val,), )


if __name__ == "__main__":
    import yaml
    config = vars(parse_args())
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    model_obj = Model_Train(config)
    model_obj.model_train()
