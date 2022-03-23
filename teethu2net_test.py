import os
from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import Teeth_dataloader
from model.teeth_U2NET import Teeth_U2NET



# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

threshold_value = 0.95
def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    predict_np[predict_np >= threshold_value] = 1
    predict_np[predict_np < threshold_value] = 0

    im = Image.fromarray(predict_np * 255).convert("RGB")
    img_name = image_name.split(os.sep)[-1]

    print(image_name)

    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), Image.ANTIALIAS)
    pb_np = np.array(imo)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    print(d_dir, imidx)
    imo.save(d_dir + imidx + ".png", quality=1000)


def main():
    # --------- 1. get image path and name ---------
    model_name = "teeth"  # teeth

    # root_dir = os.getcwd() # local에서 실행시
    root_dir = "./"  # google colab 에서 실행시 필요함.

    print("=== 01. get image path and name ===")
    image_dir = os.path.join(root_dir, "test_data", "test_images")
    print(f"image_dir:{image_dir}")

    prediction_dir = os.path.join(
        root_dir, "test_data", model_name + "_results" + os.sep
    )
    print(f"prediction_dir:{prediction_dir}")

    model_dir = os.path.join(root_dir, "saved_models", model_name, model_name + ".pth")
    model_dir = '/home/gaurav/Projects/medical/U-2-Net/saved_models/u2net/u2net_519_bce_itr_28000_train_6.231496_tar_0.884179.pth'
    print(f"model_dir:{model_dir}")
    print("==============")

    img_name_list = glob.glob(image_dir + os.sep + "*")
    print(img_name_list)
    print("==============")

    # --------- 2. dataloader ---------
    print("=== 02. dataloader ===")
    # 1. dataloader
    test_salobj_dataset = Teeth_dataloader(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(512), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # --------- 3. model define ---------
    print("=== 03. model define ===")
    print("...load U2NET---173.6 MB")
    net = Teeth_U2NET(3, 1)

    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint["model_state_dict"])
    # net.load_state_dict(torch.load(checkpoint["model_state_dict"] , map_location=torch.device('cpu')))

    print("=== 03-1. cuda is available check ===")
    if torch.cuda.is_available()==False:
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    print("=== 04. Convert an image to an image list. ===")
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available() == False:
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test, test = True)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)
        del d1, d2, d3, d4, d5, d6, d7



if __name__ == "__main__":
    main()
