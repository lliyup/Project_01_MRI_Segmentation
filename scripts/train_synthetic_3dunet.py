import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset ,DataLoader


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))#获取目录路径
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)#获取根目录
sys.path.append(os.path.join(PROJECT_ROOT))#添加根目录到系统路径

#导入自定义模块
from model.unet3d import UNet3D
from utils.metrics import dice_score, iou_score, logits_to_prediction
from losses.dice_loss import DiceLoss

#定义数据集
class SyntheticSphereDataset(Dataset):
    def __init__(self,num_samples = 40,image_size = 64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        size = self.image_size


        #创建3D坐标网格
        z,y,x = np.meshgrid(
            np.arange(size),
            np.arange(size),
            np.arange(size),
            indexing = "ij",
        )

        #随机生成球心和半径
        center_z = random.randint(size//4,size*3//4)
        center_y = random.randint(size//4,size*3//4)
        center_x = random.randint(size//4,size*3//4)
        radius = random.randint(size//8,size//4)

        distance = np.sqrt((z-center_z)**2
                           +(y-center_y)**2
                           +(x-center_x)**2
                           )
        
        mask = (distance <= radius).astype(np.int64)


        #图像 = 球体 + 随机噪声
        image = np.random.normal(loc=0.0,scale = 0.1,size = (size,size,size)).astype(np.float32)
        image += mask.astype(np.float32)*1.0


        #归一化
        image = (image - image.mean())/(image.std()+1e-8)

        image = np.expand_dims(image,axis=0)
#将numpy数组转换为PyTorch张量
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image,mask#返回图像和掩码
    #定义训练函数
def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
        model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
#迭代数据加载器
        for images,masks in dataloader:
            images = images.to(device)#图像作为input
            masks = masks.to(device)#掩码作为target

            logits = model(images)
            loss = loss_fn(logits,masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                probs = logits_to_prediction(logits)
                dice = dice_score(probs,masks,class_id = 1)
                iou = iou_score(probs,masks,class_id = 1)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

        n = len(dataloader)

        return {
            "loss": total_loss/n,
            "dice": total_dice/n,
            "iou": total_iou/n,
        }
    
def main():
        device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

        print("=" * 50)
        print("Synthetic 3D U-Net Training")
        print("=" * 50)
        print("Device:", device)


        dataset = SyntheticSphereDataset(
            num_samples = 40,
            image_size = 64,
        )

        dataloader = DataLoader(
            dataset,
            batch_size = 4,
            shuffle = True,
            num_workers =0,
        )

        model = UNet3D(
            in_channels = 1,
            num_classes = 2,
            base_channels = 16,
        ).to(device)

        loss_fn = DiceLoss(include_background = False)
        optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

        num_epochs = 10

        for epoch in range(num_epochs+1):
            metrics = train_one_epoch(
                model,
                dataloader = dataloader,
                loss_fn = loss_fn,
                optimizer = optimizer,
                device = device,
            )

            print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"Loss: {metrics['loss']:.4f} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"IoU: {metrics['iou']:.4f}"
        )
            
        save_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        os.makedirs(save_dir,exist_ok = True)
        save_path = os.path.join(save_dir, "synthetic_3dunet.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes":2,
                "base_channels":16,
            },
            save_path,
        )

        print("=" * 50)
        print("Training finished.")
        print("Model saved to:", save_path)
        print("=" * 50)


if __name__ == "__main__":
        main()
