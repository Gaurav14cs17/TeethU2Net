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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='TeethNet', help='model name')
    parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-t', '--threshold_value', default=0.95, help='threshold_value')
    parser.add_argument('-m', '--model_path',
                        default='/home/gaurav/Projects/medical/TeethU2Net/saved_models/u2net/TeethNet_159_bce_itr_22000_train_0.549952_tar_0.026701.pth')

    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_size', default=800, type=int, help='image size')

    parser.add_argument('--data_folder_name', default='test_data', help='data_folder_name')
    parser.add_argument('--num_workers', default=4, type=int)
    config = parser.parse_args()
    return config


class TestModel:
    def __init__(self, config):
        self.threshold_value = config['threshold_value']

        self.model_name = config['name']
        self.batch_size_train = config['batch_size']
        self.root_dir = ""

        self.prediction_dir = os.path.join(self.root_dir, "Results", self.model_name + "Masks_results" + os.sep)
        image_dir = os.path.join(self.root_dir, config['data_folder_name'], "Images")
        self.img_name_list = glob.glob(image_dir + os.sep + "*")
        #print(self.img_name_list)
        teeth_dataset = Teeth_dataloader(img_name_list=self.img_name_list, lbl_name_list=[],
                                         transform=transforms.Compose([RescaleT(config['input_size']),
                                                                       ToTensorLab(flag=0)]), )
        self.teeth_dataloader = DataLoader(teeth_dataset, batch_size=self.batch_size_train, shuffle=False,
                                           num_workers=config['num_workers'])
        self.model = Teeth_U2NET(config['input_channels'], config['num_classes'])
        checkpoint = torch.load(config['model_path'], map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if not torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    # normalize the predicted SOD probability map
    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def save_output(self, image_name, pred, d_dir):
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        predict_np[predict_np >= self.threshold_value] = 1
        predict_np[predict_np < self.threshold_value] = 0
        #print(predict_np.shape)
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
        print(d_dir, imidx)
        imo.save(d_dir + imidx + ".png", quality=1000)

    def model_inf(self):
        # --------- 4. inference for each image ---------
        print("=== 04. Convert an image to an image list. ===")
        for i_test, data_test in enumerate(self.teeth_dataloader):
            inputs_test = data_test["image"]
            inputs_test = inputs_test.type(torch.FloatTensor)
            if torch.cuda.is_available() == False:
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)
            d1, d2, d3, d4, d5, d6, d7 = self.model(inputs_test, test=True)
            # normalization
            pred = d1[:, 0, :, :]
            pred = self.normPRED(pred)
            # save results to test_results folder
            if not os.path.exists(self.prediction_dir):
                os.makedirs(self.prediction_dir, exist_ok=True)
            self.save_output(self.img_name_list[i_test], pred, self.prediction_dir)
            del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    config = vars(parse_args())
    model_obj = TestModel(config)
    model_obj.model_inf()
