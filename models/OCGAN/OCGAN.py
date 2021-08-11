import torch
import torchvision
import torch.nn as nn
import numpy as np


from models.OCGAN.networks import Decoder,Encoder
from models.OCGAN.networks import Discriminator_l
from models.OCGAN.networks import Discriminator_v
from models.OCGAN.networks import Classifier

from utils.utils import weights_init
from utils.visualizer import Visualizer
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

class OCgan:
    def __init__(self,opt):
        #super(OCGAN,self).__init__()
        self.opt = opt
        self.vis = Visualizer(opt)

        #networks init
        self.net_dec = Decoder(opt)
        self.net_enc = Encoder(opt)
        self.net_D_l = Discriminator_l(opt)
        self.net_D_v = Discriminator_v(opt)
        self.net_clf = Classifier(opt)

        self.net_dec.apply(weights_init)
        self.net_enc.apply(weights_init)
        self.net_D_l.apply(weights_init)
        self.net_D_v.apply(weights_init)
        self.net_clf.apply(weights_init)

        self.net_dec.cuda()
        self.net_enc.cuda()
        self.net_D_l.cuda()
        self.net_D_v.cuda()
        self.net_clf.cuda()

        #variable init
        self.input = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.l2 = torch.empty(size= (self.opt.batchsize,self.opt.latent_dim,self.opt.latent_size),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.l1 = torch.empty(size = (self.opt.batchsize,self.opt.latent_dim, self.opt.latent_size))
        self.label = torch.empty(size= (self.opt.batchsize,),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.rec_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.fake_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.real_label = torch.ones(size=(self.opt.batchsize,1),requires_grad=False).cuda()
        self.fake_label = torch.zeros(size=(self.opt.batchsize,1),requires_grad=False).cuda()


        #optimizer
        self.optimizer_enc = torch.optim.Adam(self.net_enc.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_dec = torch.optim.Adam(self.net_dec.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_D_l = torch.optim.Adam(self.net_D_l.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_D_v = torch.optim.Adam(self.net_D_v.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_clf = torch.optim.Adam(self.net_clf.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_l2 = torch.optim.Adam([{'params':self.l2}],lr =self.opt.lr, betas=(0.9,0.99))

        #criterion
        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_l1_norm = nn.L1Loss().cuda()
        self.criterion_bce = nn.BCELoss().cuda()

        


    def set_input(self, input,label):
        self.input= input.cuda()
        self.label = self.label.cuda()


    def train_ae(self):
        self.l1 = self.net_enc(self.input)
        self.rec_img = self.net_dec(self.l1)

        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()

        self.loss_AE = self.criterion_mse(self.rec_img, self.input)
        self.loss_AE.backward()

        self.optimizer_dec.step()
        self.optimizer_enc.step()

    def train(self):
    
        
        self.u = np.random.uniform(-1, 1, (self.opt.batchsize,self.opt.latent_size))
        self.l2 = torch.from_numpy(self.u).float().cuda()
        self.n = torch.randn(self.opt.batchsize, self.opt.n_channels,self.opt.isize,self.opt.isize).cuda()
        
        self.l1 = self.net_enc(self.input+self.n)
        self.rec_img = self.net_dec(self.l1)
        self.fake_img = self.net_dec(self.l2)

        loss_clf = (self.criterion_bce(self.net_clf(self.rec_img),self.real_label) \
                    + self.criterion_bce(self.net_clf(self.fake_img),self.fake_label)) / 2

        self.net_clf.zero_grad()
        loss_clf.backward(retain_graph=True)
        self.optimizer_clf.step()

        
        self.logit_real_l = self.net_D_l(self.l1)
        self.logit_fake_l = self.net_D_l(self.l2)
        self.logit_real_v = self.net_D_v(self.net_dec(self.l1))
        self.logit_fake_v = self.net_D_v(self.net_dec(self.l2))

        loss_D_l = (self.criterion_bce(self.logit_real_l,self.real_label) + self.criterion_bce(self.logit_real_l,self.fake_label))/2
        
        self.optimizer_D_l.zero_grad()
        # loss_D  = loss_D_v + loss_D_l
        self.net_dec.zero_grad()
        self.net_enc.zero_grad()
        loss_D_l.backward(retain_graph=True)
        self.optimizer_D_l.step()
        
        self.optimizer_D_v.zero_grad()
        loss_D_v = (self.criterion_bce(self.logit_real_v,self.real_label) + self.criterion_bce(self.logit_real_v,self.fake_label))/2
        loss_D_v.backward(retain_graph=True)
        self.optimizer_D_v.step()
        
        # loss_D.backward(retain_graph=True)
        
        for _ in range(self.opt.mining_iter):
            
            mining_loss = self.criterion_bce(self.net_clf(self.net_dec(self.l2)),self.real_label)
            self.optimizer_l2.zero_grad()
            mining_loss.backward(retain_graph=True)
            self.optimizer_l2.step()

        self.fake_img = self.net_dec(self.l2)
        loss_AE_l = self.criterion_bce(self.logit_real_l,self.real_label)  #+ self.criterion_bce(self.net_D_l(self.l2),self.fake_label)
        loss_AE_v = self.criterion_bce(self.net_D_v(self.fake_img),self.fake_label) #self.criterion_bce(self.logit_real_v,self.real_label) + self.criterion_bce(self.net_D_v(self.fake_img),self.fake_label)
        loss_AE_total = 10.0 * self.criterion_mse(self.rec_img ,self.input) + loss_AE_l + loss_AE_v

        # self.net_enc.zero_grad()
        # self.net_dec.zero_grad()

        loss_AE_total.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()


    def visual(self):
        self.vis.display_current_image(self.input,self.fake_img)
        
        








