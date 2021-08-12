import torch
from option import Option
from custom_dataloader import load_data
from models.OCGAN.OCGAN import OCgan
from tqdm import tqdm


if __name__=='__main__':
    opt = Option().parse()
    normal_classes = [8]
    dataloader = load_data(opt, normal_classes)
    test_loader = load_data(opt, normal_classes,check =True)
    model = OCgan(opt)

    

    for epoch in tqdm(range(opt.n_epochs),ncols=100):
        for inputs, labels in dataloader:
            model.set_input(inputs,labels)
            if epoch < 20:
                model.train_ae()
            else:
                model.train_2()

        if epoch < 20:
            model.visual()
        else:
            for fixed,label in test_loader:
                model.set_fix(fixed, label)
                break
            model.visual(True)
            model.fixed_test()
            model.visual_test()

        # model.visual()

        # for inputs, label in test_loader:
        #     model.set_input(inputs,label)
        #     model.test()
        #     model.visual(l2=True)
        #     break
        # model.visual()

