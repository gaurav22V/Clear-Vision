
"""
Image Restoration GAN: Optimized to handle dynamic input resolutions (128x128, 256x256) dynamically.
"""

import torch
import torch.nn as nn

def init_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=128):
        super().__init__()
        self.image_size = image_size
        
        # Encoder Layers 
        self.enc1 = self._make_encoder_block(in_channels, 64, norm=False)  
        self.enc2 = self._make_encoder_block(64, 128)   
        self.enc3 = self._make_encoder_block(128, 256)  
        self.enc4 = self._make_encoder_block(256, 512)  
        self.enc5 = self._make_encoder_block(512, 512)  
        self.enc6 = self._make_encoder_block(512, 512)  
        self.enc7 = self._make_encoder_block(512, 512)  
        
        self.enc8 = self._make_encoder_block(512, 512)  
        
        # Decoder Layers 
        self.dec8 = self._make_decoder_block(512, 512, dropout=True)      
        self.dec7 = self._make_decoder_block(1024, 512, dropout=True)     
        self.dec6 = self._make_decoder_block(1024, 512, dropout=True)     
        self.dec5 = self._make_decoder_block(1024, 512)                   
        self.dec4 = self._make_decoder_block(1024, 256)                   
        self.dec3 = self._make_decoder_block(512, 128)                    
        self.dec2 = self._make_decoder_block(256, 64)                     
        
        self.final_out = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
        )
        
        self.apply(init_weights)
    
    def _make_encoder_block(self, in_c, out_c, norm=True):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_c, out_c, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Forward Pass
        e1 = self.enc1(x)       
        e2 = self.enc2(e1)      
        e3 = self.enc3(e2)      
        e4 = self.enc4(e3)      
        e5 = self.enc5(e4)      
        e6 = self.enc6(e5)      
        e7 = self.enc7(e6)      
        
        if self.image_size == 256:
            e8 = self.enc8(e7)
            # Decoder Pass 
            d8 = self.dec8(e8)
            d7 = self.dec7(torch.cat([d8, e7], dim=1))
        else:
            d7 = self.dec7(torch.cat([self.dec8(e7), e7], dim=1))
            
        d6 = self.dec6(torch.cat([d7, e6], dim=1)) 
        d5 = self.dec5(torch.cat([d6, e5], dim=1)) 
        d4 = self.dec4(torch.cat([d5, e4], dim=1)) 
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) 
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) 
        
        return self.final_out(torch.cat([d2, e1], dim=1))


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator. Maps an image pair to an NxN matrix of probabilities.
    Using Spectral Normalization stabilizes the GAN adversarial optimization loop.
    """
    def __init__(self, in_channels=6):
        super().__init__()
        
        def sn_conv(in_c, out_c, k=4, s=2, p=1):
            return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p))
        
        self.feature_extractor = nn.Sequential(
            sn_conv(in_channels, 64), 
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(64, 128),          
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(128, 256),         
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(256, 512),         
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(512, 512),         
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Prediction map convolution
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)                           
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.apply(init_weights)
    
    def forward(self, condition, target):
        combined_input = torch.cat([condition, target], dim=1)
        patch_features = self.feature_extractor(combined_input)
        return self.global_pool(patch_features).view(patch_features.size(0), -1)


if __name__ == "__main__":
    for size in [128, 256]:
        print(f"\nVerifying architecture shapes for input resolution: {size}x{size}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dummy_input = torch.randn(1, 3, size, size).to(device)
        generator = UNetGenerator(image_size=size).to(device)
        discriminator = PatchDiscriminator().to(device)
        
        try:
            fake_output = generator(dummy_input)
            disc_output = discriminator(dummy_input, fake_output)
            print(f"-> Generator Output Shape:     {list(fake_output.shape)}")
            print(f"-> Discriminator Output Shape: {list(disc_output.shape)}")
            print("✓ Architecture execution verification complete!")
        except Exception as e:
            print(f"Verification failed for size {size}: {e}")