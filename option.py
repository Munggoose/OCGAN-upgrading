import argparse


class Option:
    def __init__(self):
        
        self.opt = argparse.ArgumentParser()
        #train parameter
        self.opt.add_argument("--n_epochs",type =int, default =200, help='number of epochs of training')
        self.opt.add_argument("--batchsize",type = int, default= 64, help='number of batchsize')
        self.opt.add_argument("--latent_dim",type=int, default= 32,help='dimension of latent vector' )
        self.opt.add_argument("--lr",type=float, default=0.001, help="learning rate")
        self.opt.add_argument("--train",type=bool, default=True, help="Train mode check")
        self.opt.add_argument("--device",type=str, default='cuda', help=' train device cuda or cpu')
        self.opt.add_argument("--mining_iter",type=int, default=5, help='Informative-negative mining number of iteration')

        self.opt.add_argument("--n_channels",type=int, default=1, help="number of input chanmnels")
        self.opt.add_argument("--isize",type=int,default=28 ,help="size of image")
        self.opt.add_argument("--ngf",type=int, default=32,help="number of generator features")
        self.opt.add_argument("--ndf",type=int, default=32,help='number of discriminator features')
        self.opt.add_argument("--latent_size",type=int, default=288, help="(batchsize,latent_dim,latent_size)")

        #Associate with dataset
        self.opt.add_argument("--dataset",type=str, default='MNIST',help='dataset name')
        self.opt.add_argument("--workers",type=int,default=4, help="number of workers")

        #Visdom parameter
        # self.opt.add_argument("")
    
    def parse(self):
        return self.opt.parse_args()
