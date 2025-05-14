import torch
import random
import numpy as np
import nibabel as nib
import os
import torch.nn.functional as F
import json
#from models.RMtNet import RMtNet
#from model.pretrain.segmamba import SegMamba
#from model.fine_tuning.swin_unetr import SwinUNETR
#from model.fine_tuning.ft_swinunetr.swin_unetr_pissa_dora import SwinUNETR
from model.fine_tuning.ft_swinunetr.swin_unetr_pissa1 import SwinUNETR
#from model.fine_tuning.segmamba import SegMamba


torch.manual_seed(200)
torch.cuda.manual_seed(200)
random.seed(200)
np.random.seed(200)
#model =  RMtNet()

model = SwinUNETR(img_size=64,in_channels=1,out_channels=2).cuda()

#model = UNet(1, [128, 256, 512, 768], 2, net_mode='3d')
#model = torch.nn.DataParallel(model).cuda()
 
 
# 1. 这里是加载单个模型的.pth文件：

load_file = '/home/ubuntu2204/yhx/SegMamba-main/checkpoint_pissa_dora/qkv_L/kits/train/bs=1/r=16/model_epoch_999.pth'


checkpoint = torch.load(load_file)
model.load_state_dict(checkpoint['state_dict'])



def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, cannot find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

txt_file = '/media/ubuntu2204/Elements/kits23/dataset/test.txt'
with open(txt_file, 'r') as f:
    file_names = [line.strip() for line in f.readlines()]

dice_coefficients = []
cls_one_hot_np = []

def dice_coefficient(mask, output):
    intersection = np.sum(mask * output)
    union = np.sum(mask) + np.sum(output)
    if union == 0:
        return 1.0
    else:
        return (2.0 * intersection) / union

    
average_dice = 0
correct = 0
total = 0

for file_name in file_names:
    # 构造图像文件路径和目标文件路径
    image_file = f'/media/ubuntu2204/Elements/kits23/dataset/{file_name}/final_img.nii.gz'
    target_file = f'/media/ubuntu2204/Elements/kits23/dataset/{file_name}/final_seg.nii.gz'
    #label_file = f'/home/ubuntu2204/yhx/multimask/fold_dataset/fold0/{file_name}/{file_name}_/class.txt'

    images = np.array(nib_load(image_file), dtype='float32', order='C')
    images = np.expand_dims(images, axis=-1)
    mask = images.sum(-1) > 0

    x = images[..., 0]
    y = x[mask]

    x[mask] -= y.mean()
    x[mask] /= y.std()

    images[..., 0] = x
    
    image = np.ascontiguousarray(images.transpose(3, 0, 1, 2))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    image = image.cuda()
    
    mask = model(image)

    output = F.softmax(mask, dim=1)
    output = output[0, :, :, :, :].cpu().detach().numpy()
    output = output.argmax(0)
    
    # 确保目标目录存在
    output_dir = f'/home/ubuntu2204/yhx/SegMamba-main/dataset/pred_PD/kits/train/bs=1/r=16/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果为 .nii.gz 文件
    output_img = nib.Nifti1Image(output.astype(np.float32), affine=np.eye(4))
    output_file = os.path.join(output_dir, f'{file_name}_prediction.nii')
    nib.save(output_img, output_file)

    target = np.array(nib_load(target_file), dtype='float32', order='C')
    dice_coeff = dice_coefficient(output, target)

    # Output dice coefficient for each file
    print(f"Dice Coefficient for file {file_name}: {dice_coeff}")
    average_dice += dice_coeff
    dice_coefficients.append(dice_coeff)

    
    
average_dice /= len(file_names)

print(f"Average Dice Coefficient: {average_dice}")


# 指定结果文件路径
result_file = '/home/ubuntu2204/yhx/SegMamba-main/test_pissa_dora/test50/qkv_L/kits/train/bs=1/r=16/6e-5.txt'

# 打开文件以写入结果
with open(result_file, 'w') as f:
    for file_name, dice_coeff in zip(file_names, dice_coefficients):
        # 将文件名和Dice系数写入文件
        f.write(f"File: {file_name}, Dice Coefficient: {dice_coeff}\n")

    # 写入平均Dice系数和正确率
    f.write(f"Average Dice Coefficient: {average_dice}\n")

# 输出提示信息
print(f"结果已写入文件：{result_file}")

