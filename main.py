import torch
from option import Option
from custom_dataloader import load_data
from models.OCGAN.OCGAN import OCgan

# from models.OCGAN_DCGAN_train.OCGAN import OCgan
from tqdm import tqdm
        

if __name__=='__main__':
    opt = Option().parse()
    normal_classes = [8]
    dataloader = load_data(opt,normal_classes,train=True)
    abnormal_train_loader = load_data(opt, normal_classes,train=True, check =True)
    test_loader = load_data(opt,normal_classes, train=False)
    model = OCgan(opt)
    # print(model.evaluate(dataloader))
    
    best_acc = 0.0 # best accuracy
    # for epoch in tqdm(range(opt.n_epochs),ncols=100):
    #     for inputs, labels in dataloader:
    #         model.set_input(inputs,labels)
    #         model.train_generator()
    #         model.train_discriminator()
        
    #     model.visual(True)



    for epoch in tqdm(range(opt.n_epochs),ncols=100):
        for inputs, labels in dataloader:
            model.set_input(inputs,labels)
            if epoch < 15:
                model.train_ae()
            else:
                model.train_2()

        if (epoch) % opt.test_ratio == 0:
            cur_acc = model.evaluate(test_loader, epoch)
            if cur_acc >= best_acc:
                best_acc = cur_acc
                model.save_weight(epoch)

        if epoch < 15:
            model.visual()
        else:
            for fixed,label in abnormal_train_loader:
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

