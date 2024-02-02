import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

from loaders.dataloader import meta_train_dataloader,meta_test_dataloader
from models.TFAN import TFAN
from configs import get_parser,get_logger,get_info,get_augmentation,get_score

AUG_NUM = 2
MAX_MOVE = 20

class Path_Manager(object):
    
    def __init__(self,args):
        
        dataset_dir = args.dataset_dir
        self.train = os.path.join(dataset_dir,'train')
        self.test = os.path.join(dataset_dir,'test',str(args.time_gap))
        self.val = os.path.join(dataset_dir,'val') if not args.no_val else self.test

class Meta_Trainer(object):
    
    def __init__(self, args):
        
        # Set the folder to save the records and checkpoints
        base_dir = os.path.join('./results',get_info(args))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        self.logger = get_logger(os.path.join(base_dir,'experiment.log'))
        self.writer = SummaryWriter(os.path.join(base_dir,'logs'))
        self.save_path = os.path.join(base_dir,'Model.pth')
        
        self.logger.info('All the hyper-parameters in args:')
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                self.logger.info('%s: %s' % (str(arg), str(value)))
        self.logger.info('------------------------')

        # Set args to be shareable in the class
        self.pm = Path_Manager(args)
        self.way = args.way
        self.shot = args.shot
        self.query = args.query
        self.val_epoch = args.val_epoch
        self.val_trial = args.val_trial
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.gamma = args.gamma
        self.stage = args.stage
        self.stage_size = args.stage_size
        self.total_epoch = self.stage * self.stage_size
        
        # Load meta-train set and meta-val set
        self.train_loader = meta_train_dataloader(dataset_path=self.pm.train,
                                                  way=self.way,
                                                  shots=[self.shot, self.query])
        self.val_loader = meta_test_dataloader(dataset_path=self.pm.val,
                                               way=self.way,
                                               shots=[self.shot, self.query],
                                               trial=self.val_trial)
        
        # Build Task-adaptive Feature Aligment Network
        self.model = TFAN(self.way)
        self.model = self.model.cuda()

        # Set optimizer 
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=self.lr, momentum=0.9, 
                                   weight_decay=self.weight_decay, 
                                   nesterov=True)

        # Set learning rate scheduler 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,  
                                                   gamma=self.gamma,
                                                   step_size=self.stage_size)

    """The function for the meta-train phase."""
    def train(self):
        
        # Set global count to zero
        iter_counter = 0
        best_epoch = 0
        best_val_acc = 0
     
        self.logger.info("Start training!")
        
        for epoch in range(self.total_epoch):
            
            # ================================================================================
            # Start train per epoch
            # ================================================================================
            
            # Set the model to train mode
            self.model.train()
            
            # Generate the labels for query set during meta-train updates
            target = torch.LongTensor([i // self.query for i in range(self.query * self.way)]).cuda()
            
            criterion = nn.NLLLoss().cuda()
            avg_loss = 0
            avg_acc = 0
            
            # Using tqdm to read samples from train loader per epoch
            tqdm_gen = tqdm(self.train_loader)
            
            for i, (inp,_) in enumerate(tqdm_gen):
                
                # Update global count number 
                iter_counter = iter_counter + 1
                              
                inp = get_augmentation(inp, self.way, self.shot, AUG_NUM, MAX_MOVE)         
                
                inp = inp.cuda()
                
                # Output logits for model
                # log_prediction = self.model(inp,way=self.way,shot=self.shot,query=self.query)
                log_prediction= self.model(inp,way=self.way,shot=self.shot*(1+AUG_NUM),query=self.query)
                
                # Calculate meta-train loss
                loss = criterion(log_prediction,target)
                
                # Calculate meta-train accuracy
                _,max_index = torch.max(log_prediction,1)
                acc = 100 * torch.sum(torch.eq(max_index,target)).item() / self.query / self.way
                
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch+1, loss.item(), acc))
                
                # Add loss and accuracy for the averagers
                avg_acc += acc
                avg_loss += loss.item()
                
                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update the averagers
            avg_acc = avg_acc / (i+1)
            avg_loss = avg_loss / (i+1)
            
            self.logger.info("epoch %d, avg_acc: %.3f" % (epoch+1, avg_acc))
            self.writer.add_scalar('total_loss',avg_loss, iter_counter)                  
            self.writer.add_scalar('train_acc',avg_acc, iter_counter)

            # ================================================================================
            # Start validation per val_epoch
            # ================================================================================
            if (epoch + 1) % self.val_epoch == 0:
                
                # set model to eval mode
                self.model.eval()

                with torch.no_grad(): 
                    # Generate the labels for query set during meta-val
                    target = torch.LongTensor([i // self.query for i in range(self.query * self.way)]).cuda()
                    acc_list = []
                
                    # Run meta-validation
                    for i, (inp,_) in tqdm(enumerate(self.val_loader)):
                    
                        inp = inp.cuda()
                        max_index = self.model.val_or_test(inp, way=self.way, shot=self.shot, query=self.query)

                        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / self.query / self.way
                        acc_list.append(acc)

                    val_acc, val_interval = get_score(acc_list)
   
                    self.writer.add_scalar('val_%d-way-%d-shot_acc' % (self.way, self.shot), val_acc, iter_counter)
                
                self.logger.info('val_%d-way-%d-shot_acc: %.3f\t%.3f' % (self.way, self.shot, val_acc, val_interval))
                    
                # Update best saved model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), self.save_path)
                    self.logger.info('BEST!')          

            # Update learning rate
            self.scheduler.step()
        
        self.logger.info('Training finished!')

        self.logger.info('------------------------')
        self.logger.info('The best epoch is %d/%d' % (best_epoch, self.total_epoch))
        self.logger.info('The best %d-way %d-shot val acc is %.3f' % (self.way, self.shot, best_val_acc))

    """The function for the meta-test phase."""
    def test(self):

        self.logger.info('------------------------')
        self.logger.info('Evaluating on test set:')

        with torch.no_grad():

            self.model.load_state_dict(torch.load(self.save_path))
            self.model.eval()
        
            # Load meta-test set
            test_loader = meta_test_dataloader(dataset_path=self.pm.test,
                                               way=self.way,
                                               shots=[self.shot, self.query],
                                               trial=2000)

            # Generate labels
            target = torch.LongTensor([i // self.query for i in range(self.query * self.way)]).cuda()
            acc_list = []
            
            # Start meta-test
            for i, (inp,_) in tqdm(enumerate(test_loader)):
                
                inp = inp.cuda()
                max_index = self.model.val_or_test(inp, way=self.way, shot=self.shot, query=self.query)

                acc = 100 * torch.sum(torch.eq(max_index, target)).item() / self.query / self.way
                acc_list.append(acc)

            mean, interval = get_score(acc_list)
            
            self.logger.info('%d-way-%d-shot acc: %.2f\t%.2f'%(self.way,self.shot,mean,interval))

if __name__=='__main__':

    args = get_parser()
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    trainer = Meta_Trainer(args)
    trainer.train()
    trainer.test()
