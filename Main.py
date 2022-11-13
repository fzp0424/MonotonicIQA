import argparse
import TrainModel
import scipy.io as sio
import os


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('--get_scores', type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument("--backbone", type=str, default='resnet34')
    parser.add_argument("--network", type=str, default="basecnn") #use basecnn as net
    parser.add_argument("--representation", type=str, default="BCNN")

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="./IQA_database/") #combine 6 databases
    parser.add_argument("--live_set", type=str, default="./IQA_database/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default="./IQA_database/CSIQ/")
    parser.add_argument("--bid_set", type=str, default="./IQA_database/BID/")
    parser.add_argument("--clive_set", type=str, default="./IQA_database/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="./IQA_database/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="./IQA_database/kadid10k/")

    parser.add_argument("--eval_live", type=bool, default=True)
    parser.add_argument("--eval_csiq", type=bool, default=True)
    parser.add_argument("--eval_kadid10k", type=bool, default=True)
    parser.add_argument("--eval_bid", type=bool, default=True)
    parser.add_argument("--eval_clive", type=bool, default=True)
    parser.add_argument("--eval_koniq10k", type=bool, default=True)

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')
    parser.add_argument("--train_txt", type=str, default='train.txt') # train.txt 

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.9)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)
    
    parser.add_argument("--nlm",type = bool, default= True)
    parser.add_argument("--std_modeling",type = bool, default= True)
    parser.add_argument("--continue_train",type = bool, default= False)
    parser.add_argument("--multi_gpu", type=bool, default=False)

    return parser.parse_args()


def main(cfg):
    t = TrainModel.Trainer(cfg)
    if cfg.train: 
        t.fit() #train model
    elif cfg.get_scores: 
        all_mos, all_hat, all_std, all_pstd,all_DNN_mos = t.get_scores()    # get DMOS
        scores_path = os.path.join('./scores/', ('scores' + str(cfg.split) + '.mat'))
        sio.savemat(scores_path, {'mos': all_mos, 'hat': all_hat, 'std': all_std, 'pstd': all_pstd, 'DNN_mos':all_DNN_mos})
    else:
        test_results_srcc, test_results_plcc = t.eval()                     # eval SRCC PLCC
        out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f}  KADID10K SRCC: {:.4f}' \
                  ' BID SRCC: {:.4f} CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f}'.format(test_results_srcc['live'],
                                                                                              test_results_srcc['csiq'],
                                                                                              test_results_srcc['kadid10k'],
                                                                                              test_results_srcc['bid'],
                                                                                              test_results_srcc['clive'],
                                                                                              test_results_srcc['koniq10k'])
        out_str2 = 'Testing: LIVE PLCC: {:.4f}  CSIQ PLCC: {:.4f} KADID10K PLCC: {:.4f}' \
                   ' BID PLCC: {:.4f} CLIVE PLCC: {:.4f}  KONIQ10K PLCC: {:.4f}'.format(test_results_plcc['live'],
                                                                                               test_results_plcc['csiq'],
                                                                                               test_results_plcc['kadid10k'],
                                                                                               test_results_plcc['bid'],
                                                                                               test_results_plcc['clive'],
                                                                                               test_results_plcc['koniq10k'])
        print(out_str)
        print(out_str2)


if __name__ == "__main__":

    config = parse_config()
    if config.get_scores:
        for i in range(0, 10): # 10 sessions
            config = parse_config()
            split = i + 1
            config.split = split
            config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.train = False
            config.get_scores = True
            if config.multi_gpu :
                config.ckpt = 'DataParallel-best' + str(split) +'.pt'
            else :
                config.ckpt = 'BaseCNN-best' + str(split) + '.pt'
            main(config)
    else:
        for i in range(0, 10): 
            config = parse_config()
            split = i + 1
            config.split = split
            config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)

            if config.continue_train:
                config.resume = True
                if config.multi_gpu :
                    config.ckpt = 'DataParallel-best' + str(split) +'.pt'
                else :
                    config.ckpt = 'BaseCNN-best' + str(split) + '.pt'

            config.max_epochs = config.max_epochs
            config.batch_size = config.batch_size
            main(config)








