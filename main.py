import argparse
from utils.exp_model import Exp_Model
from utils.tool import *
import datetime

set_seed(2024)

# 参数设定
parser = argparse.ArgumentParser(description='[BatterTransformer] Long Sequences Forecasting')
parser.add_argument('--model', type=str, default='BatteryTransformer', help='model of experiment, options: [BatteryTransformer ]')
parser.add_argument('--root_path', type=str, default='', help='root path of the data file') # D:\Code\MyPaper_SOH_RUL_data\\SOC_feature
parser.add_argument('--data', type=str, default='self-developed-datasets', help='data')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')  # 在下面data_parser设定
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=6, help='start token length of decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
parser.add_argument('--c_out', type=int, default=2, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=False)
parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--inverse', action='store_false', help='inverse output data', default=True)
parser.add_argument('--lradj', type=str, default='type0',help='adjust learning rate')
parser.add_argument('--weight_update', action='store_true', default=True, help='whether to Weights adaptive update')
parser.add_argument('--warm_epochs', type=int, default=5, help='warm epochs')
parser.add_argument('--cut_np', type=int, default=5, help='Noise thresholds')
parser.add_argument('--cut_fp', type=int, default=5, help='Fault thresholds')
parser.add_argument('--incremental_learn', action='store_true', default=False, help='whether to carry out small sample incremental learning')
parser.add_argument('--incremental_epoch', type=int, default=20, help='incremental learning')
parser.add_argument('--incremental_learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--session', type=int, default=10, help='incremental learning session')
parser.add_argument('--balance', type=float, default=0.05)
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=True)
parser.add_argument('--do_predict', action='store_false', help='whether to predict unseen future data', default=True)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--model_test', type=bool, default=True, help='use gpu')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
data_parser = {
    'self-developed-datasets': { 'M': [802, 802, 802], 'S': [1, 1, 1], 'MS': [800, 2, 2]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]
Exp = Exp_Model

for ii in range(args.itr):
    exp = Exp(args)
    setting = (f'{args.model}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}'
               f'_dl{args.d_layers}_df{args.d_ff}_te{args.train_epochs}_we{args.warm_epochs}_bs{args.batch_size}_pt{args.patience}'
               f'_lr{args.learning_rate}_dr{args.dropout}_lradj{args.lradj}_wu{args.weight_update}_np{args.cut_np}_fp{args.cut_fp}'
               f'_il{args.incremental_learn}_ie{args.incremental_epoch}_se{args.session}_ba{args.balance}_{ii}')
    os.makedirs(os.path.join(args.checkpoints, setting), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints, setting, 'model'), exist_ok=True)
    log = create_logger(os.path.join(args.checkpoints, setting), f'log.txt')
    log.info('Args in experiment:')
    log.info(f'{args}\n')
    exp.train(setting, log)
    destroy_logger(log)
    torch.cuda.empty_cache()
