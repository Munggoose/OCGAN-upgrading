import os
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
from torch.autograd import Variable

from models.OCGAN.evaluation import evaluate
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

class OCgan():
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
                                    dtype=torch.float32, device= torch.device(self.opt.device),requires_grad=True)
        self.l1 = torch.empty(size = (self.opt.batchsize,self.opt.latent_dim, self.opt.latent_size))
        self.label = torch.empty(size= (self.opt.batchsize,),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.rec_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.fake_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))

        self.fixed_input = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        self.fixed_rec_img = torch.empty(size= (self.opt.batchsize,self.opt.n_channels,self.opt.isize,self.opt.isize),
                                    dtype=torch.float32, device= torch.device(self.opt.device))
        
        #optimizer
        self.optimizer_enc = torch.optim.Adam(self.net_enc.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_dec = torch.optim.Adam(self.net_dec.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_D_l = torch.optim.Adam(self.net_D_l.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_D_v = torch.optim.Adam(self.net_D_v.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_clf = torch.optim.Adam(self.net_clf.parameters(), lr = self.opt.lr, betas=(0.9,0.99))
        self.optimizer_l2 = torch.optim.Adam([{'params': self.l2}],lr =self.opt.lr, betas=(0.9,0.99))

        #criterion
        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_l1_norm = nn.L1Loss().cuda()
        self.criterion_bce = nn.BCELoss().cuda()

        
    def set_fix(self,input,label):
        self.fixed_input = input.cuda()
        self.fixed_label = label.cuda()

    def set_input(self, input,label):
        self.input= input.cuda()
        self.label = label.cuda()

    def train_ae(self):

        self.net_dec.train()
        self.net_enc.train()

        self.l1 = self.net_enc(self.input)
        self.rec_img = self.net_dec(self.l1)
        
        self.net_enc.zero_grad()
        self.net_dec.zero_grad()

        self.loss_AE = self.criterion_mse(self.rec_img, self.input)
        self.loss_AE.backward()

        self.optimizer_dec.step()
        self.optimizer_enc.step()
        # print(f'\rloss_AE: {self.loss_AE}',end='')
        # self.vis.display_current_images(self.input, self.rec_img)

    def train_2(self):

        u = np.random.uniform(-1, 1, (self.opt.batchsize,self.opt.latent_size))   
        self.l2 = torch.from_numpy(u).float().cuda()
        dec_l2 = self.net_dec(self.l2)
        n = torch.randn(self.opt.batchsize,self.opt.n_channels, self.opt.isize, self.opt.isize).cuda()
        l1 = self.net_enc(self.input + n)
        logits_C_real = self.net_clf(self.net_dec(l1))
        logits_C_fake = self.net_clf(dec_l2)

        valid_logits_C = Variable(torch.Tensor(logits_C_real.shape[0], 1).fill_(1.0)
                                ,requires_grad=False).cuda()

        fake_logits_C = Variable(torch.Tensor(logits_C_fake.shape[0], 1).fill_(0.0)
                                ,requires_grad=False).cuda()

        loss_cl_real = self.criterion_bce(logits_C_real, valid_logits_C)
        loss_cl_fake = self.criterion_bce(logits_C_fake, fake_logits_C)

        loss_cl = (loss_cl_real + loss_cl_fake) /2
        self.net_clf.zero_grad()
        loss_cl.backward(retain_graph=True)
        self.optimizer_clf.step()


        #discriminator_l
        disc_l_l1 = l1.clone()
        self.net_D_l.zero_grad()
        logits_D1_l1 = self.net_D_l(disc_l_l1)
        logits_D1_l2 = self.net_D_l(self.l2)

        label_real_Dl_l1 = Variable(torch.Tensor(logits_C_real.shape[0], 1).fill_(1.0)
                                ,requires_grad=False).cuda()

        label_fake_Dl_l2 = Variable(torch.Tensor(logits_C_fake.shape[0], 1).fill_(0.0)
                                ,requires_grad=False).cuda()
        
        loss_Dl_l1  = self.criterion_bce(logits_D1_l1,label_real_Dl_l1)
        loss_Dl_l2  = self.criterion_bce(logits_D1_l2,label_fake_Dl_l2)

        loss_DL = (loss_Dl_l1+ loss_Dl_l2) /2 

        disc_v_l1 = l1.clone()
        self.net_D_v.zero_grad()
        logits_Dv_real = self.net_D_v(self.input)
        fake_img = self.net_dec(self.l2)
        logits_Dv_fake =self.net_D_v(fake_img)

        label_real_Dv = Variable(torch.Tensor(logits_C_real.shape[0], 1).fill_(1.0)
                                ,requires_grad=False).cuda()

        label_fake_Dv = Variable(torch.Tensor(logits_C_fake.shape[0], 1).fill_(0.0)
                                ,requires_grad=False).cuda()
        
        
        loss_Dv_real = self.criterion_bce(logits_Dv_real,label_real_Dv)
        loss_Dv_fake = self.criterion_bce(logits_Dv_fake,label_fake_Dv)
        loss_Dv = (loss_Dv_real + loss_Dv_fake) / 2

        self.net_D_v.zero_grad()
        loss_Dv.backward()
        self.optimizer_D_v.step()

        
        for i in range(5):
            logits_c_l2_mine = self.net_clf(self.net_dec(self.l2))
            fake_label_mine = Variable(torch.Tensor(logits_C_fake.shape[0], 1).fill_(0.0)
                                ,requires_grad=False).cuda()
            loss_mine = self.criterion_bce(logits_c_l2_mine,fake_label_mine)
            self.optimizer_l2.zero_grad()
            loss_mine.backward()
            self.optimizer_l2.step()

        self.fake_img =self.net_dec(self.l2)
        fake_ae_img = self.net_D_v(self.fake_img)
        self.rec_img = self.net_dec(l1)
        self.loss_mse = self.criterion_mse(self.rec_img,self.input)
    
        label_real_dl_ae = Variable(torch.Tensor(logits_C_fake.shape[0], 1).fill_(1.0)
                                ,requires_grad=False).cuda()
        
        self.loss_AE_l = self.criterion_bce(logits_D1_l1,label_real_dl_ae)
        logits_Dv_l2_mine = self.net_D_v(dec_l2)
        ones_logits_Dv_l2_mine = Variable(torch.Tensor(logits_Dv_l2_mine.shape[0], 1).fill_(1.0), requires_grad=False).cuda()

        self.loss_AE_v = self.criterion_bce(logits_Dv_l2_mine,ones_logits_Dv_l2_mine)

        self.loss_ae_all = 5 * self.loss_mse + self.loss_AE_v + self.loss_AE_l
        self.net_enc.zero_grad()
        self.net_dec.zero_grad()
        self.loss_ae_all.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        # print(f'\rloss_AE_v: {loss_AE_v} loss_AE_l: {loss_AE_l} loss_AE_all: {loss_ae_all}',end='')
    
    def fixed_test(self):
        # self.fake_img = self.net_dec(self.l2)
        self.fixed_rec_img = self.net_dec(self.net_enc(self.fixed_input))
        # self.net_dec.train()
        # self.net_enc.train()
    
    def evaluate(self,dataloader,epoch):
        self.net_dec.eval()
        self.net_enc.eval()

        with torch.no_grad():
            an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.opt.device)
            gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long,    device=self.opt.device)

        for i, (inputs,labels) in enumerate(dataloader):
            # self.set_input(inputs,labels)
            inputs = inputs.cuda()
            latent_i = self.net_enc(inputs)
            fake_img = self.net_dec(latent_i)
            inputs= inputs.view([self.opt.batchsize,-1]).cuda()
            fake_img = fake_img.view([self.opt.batchsize,-1])
            error = torch.mean(torch.pow((inputs - fake_img),2),dim=1)

            an_scores[i*self.opt.batchsize : i*self.opt.batchsize + len(error)] = error.reshape(error.size(0))
            gt_labels[i*self.opt.batchsize : i*self.opt.batchsize + error.size(0)] = labels.reshape(error.size(0))
        


        an_scores = (an_scores - torch.min(an_scores))/(torch.max(an_scores)-torch.min(an_scores))
        auc, thres_hold = evaluate(gt_labels, an_scores)

        self.vis.plot_current_acc(epoch,self.opt.test_ratio,auc)

        self.net_dec.train()
        self.net_enc.train()
        return auc

##
    def save_weight(self, epoch):
        if not os.path.exists(self.opt.weight_path):
            os.mkdirs(self.opt.weight_path)

        #only save decoder and encoder
        torch.save({'net_dec':self.net_dec.state_dict(),'net_enc':self.net_enc.state_dict(),'l2':self.l2},
                    os.path.join(self.opt.weight_path,f'{epoch}_weith.pt'))


    def visual(self,l2=False):
        if not l2:
            self.vis.display_current_images(self.input,self.rec_img)
        else:
            self.vis.display_current_images(self.input,self.rec_img,self.fake_img)
    
    def visual_test(self):
        self.vis.display_fixed_images(self.fixed_input,self.fixed_rec_img)
        
        








