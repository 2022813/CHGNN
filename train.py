import torch
import torch.nn as nn
from torch.optim import Adam
import os
import numpy as np
import time
import datetime
import path
import shutil
import config



args = config.parse()

# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)

use_norm = 'use-norm' if args.use_norm else 'no-norm'

#### configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name  = 'CHGNN'
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path(f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}')

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

### configure logger
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)

# load data
from data import data
from prepare import *


test_accs = []
best_val_accs, best_test_accs = [], []

resultlogger.info(args)

# load data
X, Y, G, overlappness, homogeneity = fetch_data(args)


result = []

for run in range(1, args.n_runs+1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # load data
    args.split = run


    train_idx, test_idx = data.load_split(args)
    train_idx = torch.LongTensor(train_idx).cuda()
    test_idx  = torch.LongTensor(test_idx).cuda()

    # train_idx,test_idx = data.rand_train_test_idx(Y,train_prop=0.2,valid_prop=0.25)



    inter_class_mask = data.get_class(Y,train_idx)

    # model
    model, optimizer = initialise(X, Y, G, overlappness, homogeneity, args)


    baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    baselogger.info(model)
    baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()


    from collections import Counter
    counter = Counter(Y[train_idx].tolist())
    baselogger.info(counter)
    label_rate = len(train_idx) / X.shape[0]
    baselogger.info(f'label rate: {label_rate}')

    best_test_acc, test_acc, Z = 0, 0, None
    for epoch in range(args.epochs):
        # train
        tic_epoch = time.time()
        model.train()

        optimizer.zero_grad()
        Z1,Z2,X1,X2 = model(X)

        loss_cluster = model.loss_cluster_ada_maxmargin(X1,X2,Y,train_idx)
        loss_hyperedge = model.loss_hyperedge_ada_maxmargin(Z1,Z2,Y,train_idx)
        loss_cl = args.lamda_ec * loss_hyperedge + args.lamda_cc * loss_cluster





        loss_cls = model.loss_cls(X1, X2, Y, train_idx)
        loss = loss_cl + loss_cls



        loss.backward()
        optimizer.step()

        train_time = time.time() - tic_epoch

        # eval
        model.eval()
        Z = model.test(X)
        train_acc= accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])
        result.append(test_acc)
        # log acc

        best_test_acc = max(best_test_acc, test_acc)

        baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms')


    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)



resultlogger.info(f"Average test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
