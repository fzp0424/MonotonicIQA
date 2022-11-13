from distutils.command.config import config
import os
import time
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from ImageDataset import ImageDataset
import loss as ls

#import network
from BaseCNN import BaseCNN

from Transformers import AdaptiveResize

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        
        self.config = config

        self.train_transform = transforms.Compose([
            #transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.train_batch_size = config.batch_size
        self.test_batch_size = 1
        
        if config.get_scores:
            live_str = 'live_test.txt'
            csiq_str = 'csiq_test.txt'
            kadid10k_str = 'kadid10k_test.txt'
            bid_str = 'bid_test.txt'
            clive_str = 'clive_test.txt'
            koniq10k_str = 'koniq10k_test.txt'
        
        else:
            live_str = 'live_valid.txt'
            csiq_str = 'csiq_valid.txt'
            kadid10k_str = 'kadid10k_valid.txt'
            bid_str = 'bid_valid.txt'
            clive_str = 'clive_valid.txt'
            koniq10k_str = 'koniq10k_valid.txt'



        self.train_data = ImageDataset(csv_file=os.path.join(config.trainset, 'splits2', str(config.split), config.train_txt),
                                       img_dir=os.path.join("../IQA/datasets/"),
                                       transform=self.train_transform,
                                       test= False)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=16)

        # validation set or test set configuration
        self.live_data = ImageDataset(csv_file=os.path.join(config.live_set, 'splits2', str(config.split), live_str),
                                      img_dir=os.path.join("../IQA/datasets/databaserelease2/"),
                                      transform=self.test_transform,
                                      test=True)

        self.live_loader = DataLoader(self.live_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        self.csiq_data = ImageDataset(csv_file=os.path.join(config.csiq_set, 'splits2', str(config.split), csiq_str),
                                      img_dir=os.path.join("../IQA/datasets/CSIQ/"),
                                      transform=self.test_transform,
                                      test=True)

        self.csiq_loader = DataLoader(self.csiq_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        self.kadid10k_data = ImageDataset(csv_file=os.path.join(config.kadid10k_set, 'splits2', str(config.split), kadid10k_str),
                                         img_dir=os.path.join("../IQA/datasets/kadid10k/"),
                                         transform=self.test_transform,
                                         test=True)

        self.kadid10k_loader = DataLoader(self.kadid10k_data,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=4)

        self.bid_data = ImageDataset(csv_file=os.path.join(config.bid_set, 'splits2', str(config.split), bid_str),
                                     img_dir=os.path.join("../IQA/datasets/BID/"),
                                     transform=self.test_transform,
                                     test=True)

        self.bid_loader = DataLoader(self.bid_data,
                                     batch_size=self.test_batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1)

        self.clive_data = ImageDataset(csv_file=os.path.join(config.clive_set, 'splits2', str(config.split), clive_str),
                                       img_dir=os.path.join("../IQA/datasets/ChallengeDB_release/"),
                                       transform=self.test_transform,
                                       test=True)

        self.clive_loader = DataLoader(self.clive_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)

        self.koniq10k_data = ImageDataset(csv_file=os.path.join(config.koniq10k_set, 'splits2', str(config.split), koniq10k_str),
                                       img_dir=os.path.join("../IQA/datasets/koniq-10k/"),
                                       transform=self.test_transform,
                                       test=True)

        self.koniq10k_loader = DataLoader(self.koniq10k_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=4)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        #select model
        if config.network == 'basecnn':
            self.model = BaseCNN(config)
        else:
            raise NotImplementedError("Not supported network, need to be added!")

        #use DP
        if config.multi_gpu:
            self.model = self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = self.model.to(self.device)


        self.model_name = type(self.model).__name__
        print(self.model)
        print('Use multi gpu:' + str(config.multi_gpu))

        # loss function
        self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = ls.RelativeDistLoss()
        self.loss_fn.to(self.device)
        self.loss_fn_2 = ls.norm_loss_with_normalization()
        self.loss_fn_2.to(self.device)
        
        #set optimizer
        self.lr = config.lr

        if config.multi_gpu:
            params_dict = [{'params':self.model.module.backbone.parameters(),'lr':self.lr * 0.1},
                    {'params':self.model.module.fc.parameters(),'lr':self.lr},
                    {'params':self.model.module.nlm1.parameters(),'lr':self.lr},
                    {'params':self.model.module.nlm2.parameters(),'lr':self.lr},
                    {'params':self.model.module.nlm3.parameters(),'lr':self.lr},
                    {'params':self.model.module.nlm4.parameters(),'lr':self.lr},
                    {'params':self.model.module.nlm5.parameters(),'lr':self.lr},
                    {'params':self.model.module.nlm6.parameters(),'lr':self.lr},]

        else:
            params_dict = [{'params':self.model.backbone.parameters(),'lr':self.lr * 0.1},
                    {'params':self.model.fc.parameters(),'lr':self.lr},
                    {'params':self.model.nlm1.parameters(),'lr':self.lr},
                    {'params':self.model.nlm2.parameters(),'lr':self.lr},
                    {'params':self.model.nlm3.parameters(),'lr':self.lr},
                    {'params':self.model.nlm4.parameters(),'lr':self.lr},
                    {'params':self.model.nlm5.parameters(),'lr':self.lr},
                    {'params':self.model.nlm6.parameters(),'lr':self.lr},]

        self.optimizer = torch.optim.Adam(params_dict)  


        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.train_std_loss = []
        self.test_results_srcc = {'live': [], 'csiq': [],  'kadid10k': [], 'bid': [], 'clive': [], 'koniq10k': []}
        self.test_results_plcc = {'live': [], 'csiq': [],  'kadid10k': [], 'bid': [], 'clive': [], 'koniq10k': []}
        self.srcc_imp_prop = {}
        self.loss_record = {}
        self.nlm_weight = {}
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.val_srcc_best = 0.0


        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)
        

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch_regression(epoch)
            self.scheduler.step()



    def _train_single_epoch_regression(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate backbone: {:.8f}, other layers: {:.8f}'.format(self.optimizer.param_groups[0]['lr'],self.optimizer.param_groups[1]['lr']))
        self.model.train()
        for step, sample_batched in enumerate(self.train_loader, 0):
            x, g ,std, tag = sample_batched['I'], sample_batched['mos'],sample_batched['std'],sample_batched['tag']
            x, g, tag = x.to(self.device), g.to(self.device), tag.to(self.device)
            self.optimizer.zero_grad()
        
            y, DNN_x = self.model(x,tag,istest=False)            

            self.loss_1 = self.loss_fn(y[:,0].unsqueeze(0), g[:,0].unsqueeze(0))
            self.loss_2 = self.loss_fn(y[:,1].unsqueeze(0), g[:,1].unsqueeze(0))
            self.loss_3 = self.loss_fn(y[:,2].unsqueeze(0), g[:,2].unsqueeze(0))
            self.loss_4 = self.loss_fn(y[:,3].unsqueeze(0), g[:,3].unsqueeze(0))
            self.loss_5 = self.loss_fn(y[:,4].unsqueeze(0), g[:,4].unsqueeze(0))
            self.loss_6 = self.loss_fn(y[:,5].unsqueeze(0), g[:,5].unsqueeze(0))

            loss_str_1 = ('L_sl: loss1:%.3f\t loss2:%.3f\t loss3:%.3f\t loss4:%.3f\t loss5:%.3f\t loss6:%.3f\t ')
            print(loss_str_1%(self.loss_1.data.item(),self.loss_2.data.item(),self.loss_3.data.item(),
            self.loss_4.data.item(),self.loss_5.data.item(),self.loss_6.data.item()))


            self.loss_1 += self.loss_fn_2(y[:,0].unsqueeze(0), g[:,0].unsqueeze(0))
            self.loss_2 += self.loss_fn_2(y[:,1].unsqueeze(0), g[:,1].unsqueeze(0))
            self.loss_3 += self.loss_fn_2(y[:,2].unsqueeze(0), g[:,2].unsqueeze(0))
            self.loss_4 += self.loss_fn_2(y[:,3].unsqueeze(0), g[:,3].unsqueeze(0))
            self.loss_5 += self.loss_fn_2(y[:,4].unsqueeze(0), g[:,4].unsqueeze(0))
            self.loss_6 += self.loss_fn_2(y[:,5].unsqueeze(0), g[:,5].unsqueeze(0))

            #loss average
            self.loss = (self.loss_1+self.loss_2+self.loss_3+self.loss_4+self.loss_5+self.loss_6)/6 
            #self.loss = self.loss_fn(y,g)
            loss_str_1 = ('L_all: loss1:%.3f\t loss2:%.3f\t loss3:%.3f\t loss4:%.3f\t loss5:%.3f\t loss6:%.3f\t ')
            print(loss_str_1%(self.loss_1.data.item(),self.loss_2.data.item(),self.loss_3.data.item(),
            self.loss_4.data.item(),self.loss_5.data.item(),self.loss_6.data.item()))

            self.loss.backward()
            self.optimizer.step()
        

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected,
                                examples_per_sec, duration_corrected))
            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)

        if ((epoch+1) % self.epochs_per_eval == 0):
            # evaluate in each epoch
            test_results_srcc, test_results_plcc = self.eval()
            srcc_temp = 0
            self.test_results_srcc['live'].append(test_results_srcc['live'])
            self.test_results_srcc['csiq'].append(test_results_srcc['csiq'])
            self.test_results_srcc['kadid10k'].append(test_results_srcc['kadid10k'])
            self.test_results_srcc['bid'].append(test_results_srcc['bid'])
            self.test_results_srcc['clive'].append(test_results_srcc['clive'])
            self.test_results_srcc['koniq10k'].append(test_results_srcc['koniq10k'])

            self.test_results_plcc['live'].append(test_results_plcc['live'])
            self.test_results_plcc['csiq'].append(test_results_plcc['csiq'])
            self.test_results_plcc['kadid10k'].append(test_results_plcc['kadid10k'])
            self.test_results_plcc['bid'].append(test_results_plcc['bid'])
            self.test_results_plcc['clive'].append(test_results_plcc['clive'])
            self.test_results_plcc['koniq10k'].append(test_results_plcc['koniq10k'])
            
            self.loss_record['l_total'] = loss_corrected
            self.loss_record['l1'] = self.loss_1.data.item()
            self.loss_record['l2'] = self.loss_2.data.item()
            self.loss_record['l3'] = self.loss_3.data.item()
            self.loss_record['l4'] = self.loss_4.data.item()
            self.loss_record['l5'] = self.loss_5.data.item()
            self.loss_record['l6'] = self.loss_6.data.item()

            #sum srcc of all 6 datasets, save the model which owns the best score
            #weighted_srcc = live_srcc*779 + csiq_srcc*866 + kadid10k_srcc*10125 + bid_srcc*586 + clive_srcc*1162 + koniq10k_srcc*10073 ;
            srcc_temp = test_results_srcc['live'] * 779  + test_results_srcc['csiq'] * 866  + test_results_srcc['kadid10k'] * 10125 + \
                test_results_srcc['bid'] * 586 + test_results_srcc['clive'] * 1162 + test_results_srcc['koniq10k'] * 10073
            srcc_temp = srcc_temp / (779+866+10125+586+1162+10073)

            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f}  KADID10K SRCC: {:.4f} ' \
                      'BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f} SUM_SRCC: {:.3f} '.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['kadid10k'],
                test_results_srcc['bid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'],
                srcc_temp)
               
            out_str2 = 'Testing: LIVE PLCC: {:.4f}  CSIQ PLCC: {:.4f}  KADID10K PLCC: {:.4f} ' \
                       'BID PLCC: {:.4f} CLIVE PLCC: {:.4f}  KONIQ10K PLCC: {:.4f}'.format(
                test_results_plcc['live'],
                test_results_plcc['csiq'],
                test_results_plcc['kadid10k'],
                test_results_plcc['bid'],
                test_results_plcc['clive'],
                test_results_plcc['koniq10k'])


            loss_str = 'loss_total:{:.3f} loss1:{:.3f} loss2:{:.3f} loss3:{:.3f} loss4:{:.3f} loss5:{:.3f} loss6:{:.3f} '.format(
                self.loss_record['l_total'],
                self.loss_record['l1'],
                self.loss_record['l2'],
                self.loss_record['l3'],
                self.loss_record['l4'],
                self.loss_record['l5'],
                self.loss_record['l6'])


            print(out_str)
            print(out_str2)


            record_name = '{}_{}.txt'.format(str(self.config.split),'srcc_plcc')
            record_name = os.path.join('./srcc',record_name)
            with open(record_name,'a') as f:
                f.write(out_str+'\n')
                f.write(out_str2+'\n')
                f.write(loss_str+'\n')
            
            # save the best model(max(sum(SRCC)))
            if self.val_srcc_best < srcc_temp:
                self.val_srcc_best = srcc_temp
                model_name = '{}-{}{}.pt'.format(self.model_name,'best',str(self.config.split))
                model_name = os.path.join(self.ckpt_path, model_name)
                if self.config.multi_gpu:
                    self.sd = self.model.module.state_dict()
                else:
                    self.sd = self.model.state_dict()
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.sd,
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'train_std_loss': 0,
                    'test_results_srcc': self.test_results_srcc,
                    'test_results_plcc': self.test_results_plcc,
                }, model_name)

            elif self.val_srcc_best>=srcc_temp:
                pass

        return self.loss.data.item()

    def eval(self):
        srcc = {}
        plcc = {}
        loss_ave = {}
        self.model.eval()
        tag_temp = 0

        if self.config.eval_live:
            q_mos = []
            q_hat = []
            q_loss = []
            
            for step, sample_batched in enumerate(self.live_loader, 0):
                x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
                x = x.to(self.device)
                y_bar,DNN_x = self.model(x,tag,istest=True)
                y_bar.cpu()
                #y_bar = self.model(x,tag,istest=True).to(self.device)
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())
                y_bar = y_bar.to(self.device) 
                y = y.to(self.device)
                loss_temp = self.loss_fn(y_bar, y)
                q_loss.append(loss_temp.cpu().data.numpy())
                tag_temp = tag        

            srcc['live'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['live'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            loss_ave['live'] = np.mean(q_loss)
            print('---live----',srcc['live'],plcc['live'],loss_ave['live'],tag_temp )
            
        else:
            srcc['live'] = 0
            plcc['live'] = 0

        if self.config.eval_csiq:
            q_mos = []
            q_hat = []
            q_loss = []
            for step, sample_batched in enumerate(self.csiq_loader, 0):
                x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
                x = x.to(self.device)
                y_bar,DNN_x = self.model(x,tag,istest=True)
                y_bar.cpu()
                #y_bar = self.model(x,tag,istest=True).to(self.device)
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())
                y_bar = y_bar.to(self.device) 
                y = y.to(self.device)
                loss_temp = self.loss_fn(y_bar, y)
                q_loss.append(loss_temp.cpu().data.numpy())
                tag_temp = tag 

            srcc['csiq'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['csiq'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            loss_ave['csiq'] = np.mean(q_loss)
            print('---csiq----',srcc['csiq'],plcc['csiq'],loss_ave['csiq'],tag_temp )
        else:
            srcc['csiq'] = 0
            plcc['csiq'] = 0


        if self.config.eval_kadid10k:
            q_mos = []
            q_hat = []
            q_loss = []
            for step, sample_batched in enumerate(self.kadid10k_loader, 0):
                x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
                x = x.to(self.device)
                y_bar,DNN_x = self.model(x,tag,istest=True)
                y_bar.cpu()
               # y_bar = self.model(x,tag,istest=True).to(self.device)
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())
                y_bar = y_bar.to(self.device) 
                y = y.to(self.device)
                loss_temp = self.loss_fn(y_bar, y)
                q_loss.append(loss_temp.cpu().data.numpy())
                tag_temp = tag 

            srcc['kadid10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['kadid10k'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            loss_ave['kadid10k'] = np.mean(q_loss)
            print('---kadid10k----',srcc['kadid10k'],plcc['kadid10k'],loss_ave['kadid10k'],tag_temp )
        else:
            srcc['kadid10k'] = 0
            plcc['kadid10k'] = 0

        if self.config.eval_bid:
            q_mos = []
            q_hat = []
            q_loss = []
            for step, sample_batched in enumerate(self.bid_loader, 0):
                
                x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
                x = x.to(self.device)
                y_bar,DNN_x = self.model(x,tag,istest=True)
                y_bar.cpu()
                #y_bar = self.model(x,tag,istest=True).to(self.device)
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())
                y_bar = y_bar.to(self.device) 
                y = y.to(self.device)
                loss_temp = self.loss_fn(y_bar, y)
                q_loss.append(loss_temp.cpu().data.numpy())
                tag_temp = tag 

            srcc['bid'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['bid'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            loss_ave['bid'] = np.mean(q_loss)
            print('---bid----',srcc['bid'],plcc['bid'],loss_ave['bid'],tag_temp )
        else:
            srcc['bid'] = 0
            plcc['bid'] = 0

        if self.config.eval_clive:
            q_mos = []
            q_hat = []
            q_loss = []
            for step, sample_batched in enumerate(self.clive_loader, 0):
                x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
                x = x.to(self.device)
                y_bar,DNN_x = self.model(x,tag,istest=True)
                y_bar.cpu()
                #y_bar = self.model(x,tag,istest=True).to(self.device)
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())
                y_bar = y_bar.to(self.device) 
                y = y.to(self.device)
                loss_temp = self.loss_fn(y_bar, y)
                q_loss.append(loss_temp.cpu().data.numpy())
                tag_temp = tag 

            srcc['clive'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['clive'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            loss_ave['clive'] = np.mean(q_loss)
            print('---clive----',srcc['clive'],plcc['clive'],loss_ave['clive'],tag_temp )
        else:
            srcc['clive'] = 0
            plcc['clive'] = 0


        if self.config.eval_koniq10k:
            q_mos = []
            q_hat = []
            q_loss = []
            for step, sample_batched in enumerate(self.koniq10k_loader, 0):
                x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
                x = x.to(self.device)
                y_bar,DNN_x = self.model(x,tag,istest=True)
                y_bar.cpu()
                #y_bar = self.model(x,tag,istest=True).to(self.device)
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())
                y_bar = y_bar.to(self.device) 
                y = y.to(self.device)
                loss_temp = self.loss_fn(y_bar, y)
                q_loss.append(loss_temp.cpu().data.numpy())
                tag_temp = tag 

            srcc['koniq10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['koniq10k'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            loss_ave['koniq10k'] = np.mean(q_loss)            
            print('---koniq10k----',srcc['koniq10k'],plcc['koniq10k'],loss_ave['koniq10k'],tag_temp )
        else:
            srcc['koniq10k'] = 0
            plcc['koniq10k'] = 0


        return srcc, plcc

    def get_scores(self):
        all_mos = {}
        all_hat = {}
        all_std = {}
        all_pstd = {}
        all_DNN_mos = {}
        self.model.eval()
        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        DNN_mos = []
        for step, sample_batched in enumerate(self.live_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar,DNN_x = self.model(x,tag,istest=True)
                # q_std.append(std.data.numpy())
                # q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar,DNN_x = self.model(x,tag,istest=True)
            y_bar.cpu()
            DNN_x.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
            DNN_mos.append(DNN_x.cpu().data.numpy())

        all_mos['live'] = q_mos
        all_DNN_mos['live'] = DNN_mos
        all_hat['live'] = q_hat
        all_std['live'] = q_std
        all_pstd['live'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        DNN_mos = []
        for step, sample_batched in enumerate(self.csiq_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar,DNN_x = self.model(x,tag,istest=True)
                # q_std.append(std.data.numpy())
                # q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar,DNN_x = self.model(x,tag,istest=True)
            y_bar.cpu()
            DNN_x.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
            DNN_mos.append(DNN_x.cpu().data.numpy())


        all_mos['csiq'] = q_mos
        all_DNN_mos['csiq'] = DNN_mos
        all_hat['csiq'] = q_hat
        all_std['csiq'] = q_std
        all_pstd['csiq'] = q_pstd


        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        DNN_mos = []
        for step, sample_batched in enumerate(self.kadid10k_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar,DNN_x = self.model(x,tag,istest=True)
                # q_std.append(std.data.numpy())
                # q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar,DNN_x = self.model(x,tag,istest=True)
            y_bar.cpu()
            DNN_x.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
            DNN_mos.append(DNN_x.cpu().data.numpy())

        all_mos['kadid10k'] = q_mos
        all_DNN_mos['kadid10k'] = DNN_mos
        all_hat['kadid10k'] = q_hat
        all_std['kadid10k'] = q_std
        all_pstd['kadid10k'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        DNN_mos = []
        for step, sample_batched in enumerate(self.bid_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar,DNN_x = self.model(x,tag,istest=True)
                # q_std.append(std.data.numpy())
                # q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar,DNN_x = self.model(x,tag,istest=True)
            y_bar.cpu()
            DNN_x.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
            DNN_mos.append(DNN_x.cpu().data.numpy())

        all_mos['bid'] = q_mos
        all_DNN_mos['bid'] = DNN_mos
        all_hat['bid'] = q_hat
        all_std['bid'] = q_std
        all_pstd['bid'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        DNN_mos = []
        for step, sample_batched in enumerate(self.clive_loader, 0):
            x, y, tag = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar,DNN_x = self.model(x,tag,istest=True)
                # q_std.append(std.data.numpy())
                # q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar,DNN_x = self.model(x,tag,istest=True)
            y_bar.cpu()
            DNN_x.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
            DNN_mos.append(DNN_x.cpu().data.numpy())

        all_mos['clive'] = q_mos
        all_DNN_mos['clive'] = DNN_mos
        all_hat['clive'] = q_hat
        all_std['clive'] = q_std
        all_pstd['clive'] = q_pstd

        q_mos = []
        q_hat = []
        q_std = []
        q_pstd = []
        DNN_mos = []
        for step, sample_batched in enumerate(self.koniq10k_loader, 0):
            x, y,tag  = sample_batched['I'], sample_batched['mos'], sample_batched['tag']
            x = Variable(x)
            x = x.to(self.device)

            if self.config.std_modeling:
                y_bar,DNN_x = self.model(x,tag,istest=True)
                # q_std.append(std.data.numpy())
                # q_pstd.append(torch.sqrt(var).cpu().data.numpy())
            else:
                y_bar,DNN_x = self.model(x,tag,istest=True)
            y_bar.cpu()
            DNN_x.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())
            DNN_mos.append(DNN_x.cpu().data.numpy())

        all_mos['koniq10k'] = q_mos
        all_DNN_mos['koniq10k'] = DNN_mos
        all_hat['koniq10k'] = q_hat
        all_std['koniq10k'] = q_std
        all_pstd['koniq10k'] = q_pstd

        return all_mos, all_hat, all_std, all_pstd, all_DNN_mos

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.train_std_loss = checkpoint['train_std_loss']
            self.test_results_srcc = checkpoint['test_results_srcc']
            self.test_results_plcc = checkpoint['test_results_plcc']
            if self.config.multi_gpu:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

