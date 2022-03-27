import os
from skimage import io
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import Teeth_dataloader
from model.teeth_U2NET import Teeth_U2NET
import numpy as np

from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
def metrics(preds,label):
    accuracy = accuracy_score ( preds, label )
    precision = precision_score(preds,label,average='macro')
    recall = recall_score ( preds, label, average='macro' )
    f1 = f1_score(preds, label)
    return accuracy, precision, recall, f1


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im1 = im1 > 0.5
    im2 = im2 > 0.5
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)
    return 2. * intersection.sum() / im_sum


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='TeethNet', help='model name')
    parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-t', '--threshold_value', default=0.99, help='threshold_value')
    parser.add_argument('-m', '--model_path',
                        default='/home/gaurav/Projects/medical/TeethU2Net/saved_models/u2net/TeethNet_579_bce_itr_80000_train_0.472222_tar_0.016793.pth')

    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_size', default=800, type=int, help='image size')

    parser.add_argument('--data_folder_name', default='test_data_92', help='data_folder_name')
    parser.add_argument('--onnx_flag' , default= False , type=bool , help='onnx')
    parser.add_argument('--maxtrix_flag', default=False, type=bool, help='maxtrix_flag')
    parser.add_argument('--gpu', default=False, type=bool, help='gpu')
    parser.add_argument('--num_workers', default=4, type=int)
    config = parser.parse_args()
    return config


class TestModel:
    def __init__(self, config):
        self.threshold_value = config['threshold_value']
        self.onnx_flag = config['onnx_flag']
        self.maxtrix_flag = config['maxtrix_flag']

        self.model_name = config['name']
        self.batch_size_train = config['batch_size']
        self.root_dir = ""

        self.prediction_dir = os.path.join(self.root_dir, "Results", self.model_name + "Masks_results" + os.sep)
        image_dir = os.path.join(self.root_dir, config['data_folder_name'], "Images")
        self.img_name_list = glob.glob(image_dir + os.sep + "*")

        tra_label_dir = os.path.join("gt_masks" + os.sep)
        label_ext = ".bmp"
        if config['gpu']:
            self.cuda_flag = torch.cuda.is_available()
        else:
            self.cuda_flag = False

        tra_lbl_name_list = []
        for img_path in self.img_name_list:
            img_name = img_path.split(os.sep)[-1]
            aaa = img_name.split(".")
            bbb = aaa[0:-1]
            imidx = bbb[0]
            for i in range(1, len(bbb)):
                imidx = imidx + "." + bbb[i]
            tra_lbl_name_list.append( os.path.join(self.root_dir , config['data_folder_name'] , tra_label_dir , imidx + label_ext))

        # print(len(self.img_name_list))
        #print(len(tra_lbl_name_list) , tra_lbl_name_list)
        self.data_len = len(tra_lbl_name_list)
        teeth_dataset = Teeth_dataloader(img_name_list=self.img_name_list, lbl_name_list=tra_lbl_name_list,
                                         transform=transforms.Compose([RescaleT(config['input_size']),
                                                                       ToTensorLab(flag=0)]), )

        self.teeth_dataloader = DataLoader(teeth_dataset, batch_size=self.batch_size_train, shuffle=False,
                                           num_workers=config['num_workers'])
        self.model = Teeth_U2NET(config['input_channels'], config['num_classes'])
        checkpoint = torch.load(config['model_path'], map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.cuda_flag:
            self.model.cuda()
        self.model.eval()

    # normalize the predicted SOD probability map
    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def save_output(self, image_name, pred, d_dir ,i_test =0 ):
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        predict_np[predict_np >= self.threshold_value] = 1
        predict_np[predict_np < self.threshold_value] = 0
        # print(predict_np.shape)
        im = Image.fromarray(predict_np * 255).convert("RGB")
        img_name = image_name.split(os.sep)[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1], image.shape[0]), Image.ANTIALIAS)
        # pb_np = np.array(imo)
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        print(" Image number" ,i_test, d_dir, imidx)
        imo.save(d_dir + imidx + ".png", quality=1000)

    def model_inf(self  ):
        # --------- 4. inference for each image ---------
        print("=== 04. Convert an image to an image list. ===")
        FP, FN, TP, TN , Acc = 0,0,0,0,0

        for i_test, data_test in enumerate(self.teeth_dataloader):
            inputs_test = data_test["image"]
            gt_labels = data_test["label"]
            inputs_test = inputs_test.type(torch.FloatTensor)
            if self.cuda_flag:
                inputs_test = Variable(inputs_test.cuda())
                gt = Variable(gt_labels.cuda(), requires_grad=False)
            else:
                inputs_test = Variable(inputs_test)
                gt = Variable(gt_labels, requires_grad=False)

            if self.onnx_flag:
                torch.onnx.export(self.model,  # model being run
                                  inputs_test,  # model input (or a tuple for multiple inputs)
                                  "teeth_net.onnx",
                                  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=10,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  input_names=['input'],  # the model's input names
                                  output_names=['output'],  # the model's output names
                                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                                'output': {0: 'batch_size'}})
                print('onnx done')
                exit()

            d1, d2, d3, d4, d5, d6, d7 = self.model(inputs_test, test=True)
            # normalization
            pred = d1[:, 0, :, :]

            # if  self.maxtrix_flag :
            #     predict = pred.squeeze()
            #     predict_np = predict.cpu().data.numpy()
            #     predict_np[predict_np >= self.threshold_value] = 1
            #     predict_np[predict_np < self.threshold_value] = 0
            #     # print(predict_np.shape)
            #     pred = Image.fromarray(predict_np * 255).convert("RGB")
            #
            #     Acc += accuracy_score(pred.cpu().data.numpy(), gt.cpu().data.numpy())
            #     mat =  numeric_score(pred.cpu().data.numpy(), gt.cpu().data.numpy())
            #     FP += mat[0]
            #     FN += mat[1]
            #     TP += mat[2]
            #     TN += mat[3]
            pred = self.normPRED(pred)
            # save results to test_results folder
            if not os.path.exists(self.prediction_dir):
                os.makedirs(self.prediction_dir, exist_ok=True)
            self.save_output(self.img_name_list[i_test], pred, self.prediction_dir , i_test )
            del d1, d2, d3, d4, d5, d6, d7
        # if  self.maxtrix_flag:
        #     print('Acc : %.2f' % (Acc / self.data_len))
        #     print('FP  :  %.2f' % (FP / self.data_len))
        #     print('FN  :  %.2f' % (FN / self.data_len))
        #     print('TP  :  %.2f' % (TP / self.data_len))
        #     print('TN  :  %.2f' % (TN / self.data_len))


if __name__ == "__main__":
    config = vars(parse_args())
    model_obj = TestModel(config)
    model_obj.model_inf()
