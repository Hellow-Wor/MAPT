import math
import os
import time

import numpy as np
import torch
from Demos.win32ts_logoff_disconnected import session
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.data import session_data
from data.data_loader import Dataset_self
from utils.tool import *
from models.model import BatteryTransformer
from scipy.stats import norm
from data import *

class Exp_Model(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'BatteryTransformer': BatteryTransformer,
        }
        self.data_dict = {
            'self-developed-datasets': Dataset_self,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def _build_model(self):
        model = self.model_dict[self.args.model](
            enc_in=self.args.enc_in,
            dec_in=self.args.dec_in,
            c_out=self.args.c_out,
            session=self.args.session,
            label_len=self.args.label_len,
            pred_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_layers=self.args.d_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            attn=self.args.attn,
            activation=self.args.activation,
            inverse=self.args.inverse,
            output_attention=self.args.output_attention,
            distil=self.args.distil,
            device=self.device
        ).float()
        return model
    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')
        return criterion
    def _get_data(self, flag, battery_list, session):
        Data = self.data_dict[self.args.data]
        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = 1
        elif flag == 'pred':
            shuffle_flag = False; drop_last = False; batch_size = 1
        else:  # train / vali
            shuffle_flag = True; drop_last = False; batch_size = self.args.batch_size
        data_set = Data(
            root_path=self.args.root_path,
            battery_list=battery_list,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.enc_in,
            session=session,
        )
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=self.args.num_workers, drop_last=drop_last, pin_memory=False)
        return data_set, data_loader
    def train(self, setting, log):
        path = os.path.join(self.args.checkpoints, setting)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for session in range(self.args.session):
            log.info(f'--------------------------------------------------session{session} start training--------------------------------------------------------------------')
            log.info(f"training session {session}, train battery list : {session_data[f'session_{session}']['train']}")
            train_set, train_loader = self._get_data(flag='train', battery_list=session_data[f'session_{session}']['train'], session=session)
            log.info(f'train len {len(train_set)}')
            best_model_path = os.path.join(path, 'model', f'model_session{session}.pth')
            # The noise-aware reweighting module
            indicies = np.arange(len(train_loader.dataset))
            memory_module = dict([(k,[]) for k in indicies])

            if session == 0:
                vali_set, vali_loader = self._get_data(flag='vali', battery_list=session_data[f'session_{session}']['vali'], session=session)
                log.info(f"vali data {session_data[f'session_{session}']['vali']}")
                log.info(f'vali len {len(vali_set)}')
                early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
                for epoch in range(self.args.train_epochs):
                    log.info(f'---------------------------------------------------------------------------------------------------------------------------------------------')
                    epoch_time = time.time()
                    iter_count = 0
                    train_loss = []
                    self.model.train()
                    # warmup the noise-aware reweighting module training epoch -> the noise-aware reweighting module training epoch
                    if epoch >= self.args.warm_epochs:
                        epoch_weight = [(1+i)/self.args.train_epochs for i in range(epoch)]
                        instance_mean = {k: np.mean(np.array(v)*epoch_weight) for k, v in sorted(memory_module.items(), key=lambda item: item[1])}
                        mu = np.mean(list(instance_mean.values()))
                        sd = np.std(list(instance_mean.values()))
                        gaussian_norm = norm(mu, sd)
                        np_bound = mu-self.args.cut_np*sd
                        fp_bound = mu+self.args.cut_fp*sd
                        np_index = [k for k in instance_mean.keys() if instance_mean[k]<=np_bound]
                        fp_index = [k for k in instance_mean.keys() if instance_mean[k]>=fp_bound]

                    for ii, (batch_x, batch_y, index) in enumerate(train_loader):
                        iter_count += 1
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float()
                        decoder_input = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        decoder_input = torch.cat([batch_y[:, :self.args.label_len, :], decoder_input], dim=1).float().to(self.device)
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    output, attn = self.model(batch_x, decoder_input)
                                else:
                                    output = self.model(batch_x, decoder_input)
                        else:
                            if self.args.output_attention:
                                output, attn = self.model(batch_x, decoder_input)
                            else:
                                output = self.model(batch_x, decoder_input)
                        all_pre = output
                        output = output[:, :, :, session].squeeze(-1)
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        loss = criterion(output, batch_y)
                        loss = loss.mean(dim=[1, 2])

                        # The virtual prototype adaptive module
                        if self.args.incremental_learn:
                            pseudo_label = torch.mean(all_pre[:, :, :, 1:], dim=-1)
                            pseudo_label = pseudo_label.unsqueeze(-1).repeat(1,1,1,self.args.session-1)
                            loss1 = criterion(all_pre[:, :, :, 1:], pseudo_label)
                            loss = loss + self.args.balance*loss1.mean(dim=[1, 2, 3])

                        #* update the noise-aware reweighting module
                        for j in range(len(index)):
                            memory_module[index[j].cpu().item()].append(loss[j].cpu().item())
                        #*  the noise-aware reweighting module re-weighting process
                        if epoch >= self.args.warm_epochs and self.args.weight_update:
                            l = loss.detach().cpu()
                            w = gaussian_norm.pdf(l)
                            for j in range(len(index)):
                                _id = index[j].cpu().item()
                                if _id in np_index or _id in fp_index:
                                    loss[j] *= w[j]
                        loss = torch.mean(loss)
                        train_loss.append(loss.item())

                        if (ii+1) % 100==0:
                            speed = (time.time()-epoch_time) / iter_count
                            left_time = speed*((self.args.train_epochs - epoch)*len(train_loader) - ii)
                            log.info(f"\titers: {ii + 1}, epoch: {epoch + 1} | batch loss: {loss.item():.10g} | speed: {speed:.4f}s/iter; left time: {left_time:.5g}s")
                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()

                    # vali
                    vali_loss = self.vali(vali_loader, criterion, session=session)
                    log.info(f"Base Training Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {np.average(train_loss):.8g} Vali Loss: {vali_loss:.8g} ")
                    early_stopping(vali_loss, self.model, best_model_path, log)
                    if early_stopping.early_stop:
                        log.info("Early stopping")
                        break
                    adjust_learning_rate(model_optim, epoch+1, self.args, log)

            if session >= 1:
                log.info(f'The base model of incremental learning is model_session{session-1}.pth')
                self.model.load_state_dict(torch.load(os.path.join(path, 'model', f'model_session{session-1}.pth')))
                self.model.train()
                self.incremental_train(train_loader, best_model_path, session, log=log)
            log.info(f'--------------------------------------------------session{session} start testing---------------------------------------------------------------------')
            log.info(f"test data {session_data[f'session_{session}']['test']}")
            self.test(test_battery=session_data[f'session_{session}']['test'], model_path=best_model_path, result_path=path, session=session, log=log)
        return self.model
    def vali(self, vali_loader, criterion, session=0):
        self.model.eval()
        vali_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                decoder_input = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                decoder_input = torch.cat([batch_y[:, :self.args.label_len, :], decoder_input], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            output, attn = self.model(batch_x, decoder_input)
                        else:
                            output = self.model(batch_x, decoder_input)
                else:
                    if self.args.output_attention:
                        output, attn = self.model(batch_x, decoder_input)
                    else:
                        output = self.model(batch_x, decoder_input)
                all_output = output.detach().cpu()
                output = output[:, :, :, session].squeeze(-1)
                output = output[:, :, :].detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, :]
                loss = criterion(output, batch_y)
                loss = loss.mean(dim=[1, 2])
                if self.args.incremental_learn:
                    pseudo_label = torch.mean(all_output[:, :, :, 1:], dim=-1)
                    pseudo_label = pseudo_label.unsqueeze(-1).repeat(1,1,1,self.args.session-1)
                    loss1 = criterion(all_output[:, :, :, 1:], pseudo_label)
                    loss = loss + self.args.balance*loss1.mean(dim=[1, 2, 3])
                loss = loss.mean()
                vali_loss.append(loss.item())
        vali_loss = np.average(vali_loss)
        self.model.train()
        return vali_loss
    def test(self, test_battery, model_path, result_path, session, log):
        log.info(f'loading model {os.path.basename(model_path)}')
        for id in test_battery:
            test_data, test_loader = self._get_data(flag='test', battery_list=[id], session=session)
            self.model.load_state_dict(torch.load(model_path))
            preds = []
            trues = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, index) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    decoder_input = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    decoder_input = torch.cat([batch_y[:, :self.args.label_len, :], decoder_input], dim=1).float().to(self.device)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                output, attn = self.model(batch_x, decoder_input)
                            else:
                                output = self.model(batch_x, decoder_input)
                    else:
                        if self.args.output_attention:
                            output, attn = self.model(batch_x, decoder_input)
                        else:
                            output = self.model(batch_x, decoder_input)
                    if self.args.incremental_learn:
                        output = output[:, :, :, session].squeeze(-1)
                    else:
                        output = output[:, :, :, 0].squeeze(-1)
                    output = output[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    output = output.detach().cpu().numpy()
                    preds.append(output)
                    trues.append(batch_y)
            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            log.info(f'{id} mae:{mae:.7g} mse:{mse:.7g} rmse:{rmse:.7g}, mape:{mape:.7g}')
            np.savez(os.path.join(result_path, f'{id}.npz'), preds=preds, trues=trues)
        return

    def incremental_train(self, train_loader, save_path, session, log):
        for name, param in self.model.named_parameters():
            if 'projection2' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        best_loss = np.inf
        criterion = nn.MSELoss()
        lr = self.args.incremental_learning_rate
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.incremental_epoch):
            epoch_loss = []
            for batch_x, batch_y, index in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                decoder_input = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                decoder_input = torch.cat([batch_y[:, :self.args.label_len, :], decoder_input], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            output, attn = self.model(batch_x, decoder_input)
                        else:
                            output = self.model(batch_x, decoder_input)
                else:
                    if self.args.output_attention:
                        output, attn = self.model(batch_x, decoder_input)
                    else:
                        output = self.model(batch_x, decoder_input)
                if self.args.incremental_learn:
                    output = output[:, :, :, session].squeeze(-1)
                else:
                    output = output[:, :, :, 0].squeeze(-1)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = criterion(output, batch_y)
                epoch_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            epoch_loss = np.average(epoch_loss)
            log.info(f"Incremental Training Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {epoch_loss:.10g} ")
            if epoch_loss < best_loss:
                log.info(f"Save model | loss decreased ({best_loss:.10g} --> {epoch_loss:.10g}). Saving model ...")
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), save_path)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (0.5 ** ((epoch+1)//4))
            log.info('Updating learning rate to {}'.format(lr * (0.5 ** ((epoch+1)//4))))
