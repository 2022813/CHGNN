# CHGNN





## Getting Started

### Prerequisites

Our code requires Python>=3.6. 

We recommend using a virtual environment and installing the newest versions of  [Pytorch](https://pytorch.org/) and [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric).


You also need these additional packages:

* scipy
* path
* tqdm


## Datasets

co-authorship hypergraphs
* Cora
* DBLP

co-citation hypergraphs
* Pubmed
* Citeseer
* Cora

other hypergraphs
*20newsgroup
*ModelNet40
*NTU2012
*Mushroom

## Baselines
UniGNN, HyperGCN, HyperSAGE, and HGNN  can be found   at [https://github.com/OneForward/UniGNN](https://github.com/OneForward/UniGNN).

AllSetTransformer can be found at [https://github.com/jianhao2016/AllSet](https://github.com/jianhao2016/AllSet).

SimGRACE can be found at [https://github.com/junxia97/SimGRACE](https://github.com/junxia97/SimGRACE).

## Semi-supervised Hypernode Classification

```sh
python train.py --data=coauthorship --dataset=cora 
```

You should probably see final accuracies like the following.  

`Average test accuracy: 76.79657001495361 ± 1.01541351159170083`


## Usage


```
usage: CHGNN [-h] [--data DATA] [--dataset DATASET]
             [--node_dropping_rate NODE_DROPPING_RATE]
             [--edge_perturbation_rate EDGE_PERTURBATION_RATE]
             [--lamda_ec LAMDA_EC] [--lamda_cc LAMDA_CC]
             [--first-aggregate FIRST_AGGREGATE]
             [--second-aggregate SECOND_AGGREGATE] [--use-norm]
             [--activation ACTIVATION] [--nlayer NLAYER] [--nhid NHID]
             [--nhead NHEAD] [--nproj NPROJ] [--dropout DROPOUT]
             [--input-drop INPUT_DROP] [--attn-drop ATTN_DROP] [--lr LR]
             [--wd WD] [--epochs EPOCHS] [--n-runs N_RUNS] [--gpu GPU]
             [--seed SEED] [--patience PATIENCE] [--nostdout] [--split SPLIT]
             [--out-dir OUT_DIR] [--cut_off_node CUT_OFF_NODE]
             [--cut_off_edge CUT_OFF_EDGE] [--tau TAU] [--epcc EPCC]
             [--epec EPEC]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data name (coauthorship/cocitation) (default:
                        coauthorship)
  --dataset DATASET     dataset name (e.g.: cora/dblp for coauthorship,
                        cora/citeseer/pubmed for cocitation) (default: cora)
  --node_dropping_rate NODE_DROPPING_RATE
                        node dropping rate (default: 0.1)
  --edge_perturbation_rate EDGE_PERTURBATION_RATE
                        edge perturbation rate (default: 0.1)
  --lamda_ec LAMDA_EC   loss rate (default: 0.1)
  --lamda_cc LAMDA_CC   loss rate (default: 1)
  --first-aggregate FIRST_AGGREGATE
                        aggregation for hyperedge h_e: max, sum, mean (default:
                        mean)
  --second-aggregate SECOND_AGGREGATE
                        aggregation for node x_i: max, sum, mean (default: sum)
  --use-norm            use norm in the final layer (default: False)
  --activation ACTIVATION
                        activation layer between UniConvs (default: relu)
  --nlayer NLAYER       number of hidden layers (default: 2)
  --nhid NHID           number of hidden features, note that actually it's #nhid
                        x #nhead (default: 8)
  --nhead NHEAD         number of conv heads (default: 8)
  --nproj NPROJ         number of projection (default: 16)
  --dropout DROPOUT     dropout probability after UniConv layer (default: 0.6)
  --input-drop INPUT_DROP
                        dropout probability for input layer (default: 0.6)
  --attn-drop ATTN_DROP
                        dropout probability for attentions in UniGATConv
                        (default: 0.6)
  --lr LR               learning rate (default: 0.01)
  --wd WD               weight decay (default: 0.0005)
  --epochs EPOCHS       number of epochs to train (default: 2000)
  --n-runs N_RUNS       number of runs for repeated experiments (default: 1)
  --gpu GPU             gpu id to use (default: 0)
  --seed SEED           seed for randomness (default: 1)
  --patience PATIENCE   early stop after specific epochs (default: 200)
  --nostdout            do not output logging to terminal (default: False)
  --split SPLIT         choose which train/test split to use (default: 1)
  --out-dir OUT_DIR     output dir (default: runs/test)
  --cut_off_node CUT_OFF_NODE
                        node dropping cut-off probability (default: 0.8)
  --cut_off_edge CUT_OFF_EDGE
                        edge perturbation cut-off probability (default: 0.8)
  --tau TAU             tau (default: 0.5)
  --epcc EPCC           epcc (default: 0.2)
  --epec EPEC           epec (default: 1.4)


```






