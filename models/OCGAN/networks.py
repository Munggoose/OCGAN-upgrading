import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import math

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from custom_dataloader import load_data
from option import Option


class Encoder(nn.Module):
    """DCGAN ENCODER

    Args:
        nn ([type]): [description]
    """

    def __init__(self,opt):
        super(Encoder,self).__init__()
        def conv_module(_in,_out):
            layers = []
            layers.append(nn.Conv2d(_in,_out,3,1,1,bias=True))
            layers.append(nn.BatchNorm2d(_out))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        
        self.input_layer = nn.Sequential(
            nn.Conv2d(opt.n_channels, opt.ndf,4,2,1),
            nn.Conv2d(opt.ndf, opt.ndf, 4,2,1),
            nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=False))

        # size 유지
        self.main = nn.Sequential(
            *conv_module(opt.ndf,opt.ndf*2),
            *conv_module(opt.ndf*2,opt.ndf*2),
        )
        
        #
        self.out_layer = nn.Sequential(
            nn.Conv2d(opt.ndf*2,opt.ndf*2, 3,1,0),
            nn.Tanh(), 
            nn.Conv2d(opt.ndf*2,opt.latent_dim,3,1,0),
            nn.Tanh()
        )


    def forward(self, input):
        output = self.input_layer(input)
        # print(output.shape)
        output = self.main(output)
        # print(output.shape)
        output = self.out_layer(output)
        # print(output.shape)
        output = output.view(output.size(0),-1)
        # print(output.shape)
        return output


class Decoder_2( nn.Module):
    """DCGAN DECODER

    Args:
        nn ([type]): [description]
    """
    def __init__(self, opt):
        self.opt= opt
        super(Decoder, self).__init__()
        def convT(_in,_out):

            layer = []
            # layer.append(nn.Upsample(scale_factor=2, mode = 'nearest'))
            layer.append(nn.ConvTranspose2d(_in,_out, 3,1,1))
            layer.append(nn.BatchNorm2d(_out))
            layer.append(nn.ReLU())

            return layer

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_dim, opt.ngf,3,1,0),
            nn.ConvTranspose2d(opt.ngf, opt.ngf,3,1,0),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            *convT(opt.ngf,opt.ngf*2),
            *convT(opt.ngf*2,opt.ngf)
        )
    
        self.last_layer = nn.Sequential(
                            nn.ConvTranspose2d(opt.ngf, opt.ngf,4,2,1),
                            nn.BatchNorm2d(opt.ngf),
                            nn.ReLU(),
                            nn.ConvTranspose2d(opt.ngf, opt.n_channels,4,2,1),
                            nn.Sigmoid()
                            )


    def forward(self, input):
        output = input.view(input.size(0),self.opt.latent_dim,3,3)
        output = self.input_layer(output)
        # print(output.shape)
        output = self.main(output)
        # print(output.shape)
        output = self.last_layer(output)
        return output


class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self,opt):
        super(Decoder, self).__init__()
        self.conv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.ConvTranspose2d(32, 32, 3, padding=3//2)
        # self.activation = nn.Tanh()
        self.conv3 = nn.ConvTranspose2d(32, 32, 3, padding=3//2)
        self.batch_norm_1 = nn.BatchNorm2d(32)  

        self.conv4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.ConvTranspose2d(32, 64, 3, padding=3//2)
        self.conv6 = nn.ConvTranspose2d(64, 64, 3, padding=3//2)
        self.batch_norm_2 = nn.BatchNorm2d(64)    
        
        self.conv7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.ConvTranspose2d(64, 64, 3)
        self.conv9 = nn.ConvTranspose2d(64, 64, 3)
        self.batch_norm_3 = nn.BatchNorm2d(64)    

        self.conv10 = nn.ConvTranspose2d(64, 1, 3, padding=3//2)    

    def forward(self, input):
        # print('l2_input:',input.size(0))
        output = input.view(input.size(0),32,3,3)
        # print('l2_reshape:',output.shape)
        output = self.conv1(output) 
        # print('l2_output1:',output.shape)
        output = torch.tanh(self.conv2(output))
        # print('l2_output2:',output.shape)
        output = torch.tanh(self.conv3(output))
        output = self.batch_norm_1(output)
        # print('l2_output3:',output.shape)

        output = self.conv4(output)
        # print('l2_output4:',output.shape)
        output = torch.tanh(self.conv5(output))
        # print('l2_output5:'/,output.shape)
        output = torch.tanh(self.conv6(output))
        output = self.batch_norm_2(output)
        # print('l2_output6:',output.shape)

        output = self.conv7(output)
        # print('l2_output7:',output.shape)
        output = torch.tanh(self.conv8(output)) 
        # print('l2_output8:',output.shape)
        output = torch.tanh(self.conv9(output))
        output = self.batch_norm_3(output)
        # print('l2_outpu/t9:',output.shape)
        # output = output.permute((0, 2, 3, 1))
        # output = output.contiguous().view(-1, 4 * 4 * 64) 
        output = torch.sigmoid(self.conv10(output))   
        # print('l2_output10:',output.shape)
        return output

class Discriminator_l(nn.Module):
    """Latent Discriminator

    Args:
        nn ([type]): [description]
    """
    def __init__(self,opt):
        super(Discriminator_l,self).__init__()
        self.li_1 = nn.Linear(288, 128)
        self.batch_1 = nn.BatchNorm1d(128)

        self.li_2 = nn.Linear(128, 64)
        self.batch_2 = nn.BatchNorm1d(64)

        self.li_3 = nn.Linear(64, 32)
        self.batch_3 = nn.BatchNorm1d(32)

        self.li_4 = nn.Linear(32, 16)
        self.batch_4 = nn.BatchNorm1d(16)

        self.li_5 = nn.Linear(16, 1)

    def forward(self, input):
        output = input.view(input.size(0),-1)
        output = torch.relu(self.batch_1(self.li_1(output)))
        output = torch.relu(self.batch_2(self.li_2(output)))
        output = torch.relu(self.batch_3(self.li_3(output)))
        output = torch.relu(self.batch_4(self.li_4(output)))
        output = torch.sigmoid(self.li_5(output))

        return output

class Discriminator_v(nn.Module):
    """
    DISCRIMINATOR vision  NETWORK
    """
    def __init__(self,opt):
        super(Discriminator_v, self).__init__()
        self.conv1 = nn.Conv2d(1,16,4,stride=2,padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(16,16,4,stride=2,padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16,16,4,stride=2,padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(16)
                
        self.conv4 = nn.Conv2d(16,1,4,stride=2,padding=1)
        self.conv5 = nn.AdaptiveAvgPool2d(output_size=1)
        # self.sigmoid = torch.sigmoid()

    def forward(self, input):

        output = self.batch_norm_1(self.conv1(input))
        # print('check',output.shape)

        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = torch.sigmoid(self.conv5(output))
        output = output.view(output.size(0), -1)

        return output

class Classifier(nn.Module):

    def __init__(self,opt):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,4,stride=2,padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(32,64,4,stride=2,padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,4,stride=2,padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
                
        self.conv4 = nn.Conv2d(64,opt.n_channels,4,stride=2,padding=1)
        self.conv5 = nn.AdaptiveAvgPool2d(output_size=1)
        # self.sigmoid = torch.sigmoid()

    def forward(self, input):

        output = self.batch_norm_1(self.conv1(input))
        # print('check',output.shape)

        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = torch.sigmoid(self.conv5(output))
        output = output.view(output.size(0), -1)

        return output

if __name__ == '__main__':
    opt = Option().parse()
    dataloader = load_data(opt,[1])
    enc = Encoder(opt)
    dec = Decoder(opt)
    dec_l = Discriminator_l(opt)
    dec_v = Discriminator_v(opt)
    clf = Classifier(opt)

    for input, label in dataloader:
        output = enc(input)
        dec_output = dec(output)
        dec_l_output = dec_l(output)
        dec_v_output = dec_v(input)
        clf_output = clf(input)


        print(output.shape)
        print(dec_output.shape)
        print(dec_l_output.shape)
        print(dec_v_output.shape)
        print(clf_output.shape)
        exit()
        
        print(output.shape)
        

        print(output.shape)
        output = dec(output)
        print(output.shape)
        exit()


