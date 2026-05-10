import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv3D,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size = 3,
                padding = 1,
                bias = False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace = True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size = 3,
                padding = 1,
                bias = False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace = True)
        )
    def forward(self,x):
        return self.block(x)
    

class UNet3D(nn.Module):
    def __init__(self,in_channels=1,num_classes = 2,base_channels = 16):
        super().__init__()
        #encoder
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv3D(base_channels,base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv3D(base_channels*2,base_channels*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        #botteleneck
        self.bottleneck = DoubleConv3D(base_channels*4,base_channels*8)

        #decoder
        self.up3 = nn.ConvTranspose3d(
            base_channels*8,
            base_channels*4,
            kernel_size = 2,
            stride = 2,
        )
        self.dec3 = DoubleConv3D(base_channels*8,base_channels*4)
        self.up2 = nn.ConvTranspose3d(
            base_channels*4,
            base_channels*2,
            kernel_size = 2,
            stride = 2,
        )
        self.dec2 = DoubleConv3D(base_channels*4,base_channels*2)
        self.up1 = nn.ConvTranspose3d(
            base_channels*2,    
            base_channels,
            kernel_size = 2,    
            stride = 2,
        )
        self.dec1 = DoubleConv3D(base_channels*2,base_channels)
        self.conv_last = nn.Conv3d(base_channels,num_classes,kernel_size=1)
    def forward(self,x):
        #encoder
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)

        #bottleneck
        bottleneck = self.bottleneck(p3)

        #decoder+skip connection
        up3 = self.up3(bottleneck)
        dec3 = torch.cat((up3,enc3),dim=1)
        dec3 = self.dec3(dec3)

        up2 = self.up2(dec3)
        dec2 = torch.cat((up2,enc2),dim=1)
        dec2 = self.dec2(dec2)

        up1 = self.up1(dec2)
        dec1 = torch.cat((up1,enc1),dim=1)
        dec1 = self.dec1(dec1)
        logits = self.conv_last(dec1)


        return logits
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()else"cpu")
    model = UNet3D(
        in_channels = 1,
        num_classes = 2,
        base_channels = 16,
    ).to(device)
    x = torch.randn(1,1,64,64,64).to(device)
    
    with torch.no_grad():
        output = model(x)

        print("=" * 50)
        print("3D U-Net Forward Test")
        print("=" * 50)
        print("Device:", device)
        print("Input shape:", x.shape)
        print("Output shape:", output.shape)
        print("Trainable parameters:", count_parameters(model))
        print("=" * 50)