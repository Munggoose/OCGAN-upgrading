import torch
from option import Option
from custom_dataloader import load_data
from models.OCGAN.OCGAN import OCgan
from tqdm import tqdm


if __name__=='__main__':
    opt = Option().parse()
    normal_classes = [8]
    dataloader = load_data(opt, normal_classes)
    model = OCgan(opt)


    for epoch in range(opt.n_epochs):
        for inputs, labels in tqdm(dataloader):
            model.set_input(inputs,labels)
            model.train()
        model.visual()

