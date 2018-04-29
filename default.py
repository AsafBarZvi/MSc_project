import json
from easydict import EasyDict as edict

default_config = {
    "gpu"               : 3,                            #gpu device used
    "data_dir"          : ".",                          #data dir containing train.txt test.txt
    "batch_size"        : 50,                           #batch size
    "weight_decay"      : 0.0005,                       #decay on weights
#    "l2_bias_decay"     : 0.000001,                    #l2 decay on biases
    "snapdir"           : "./snaps",                    #path to save snapshots
    "epochs"            : 20,                           #number of epoches
    "logdir"            : "logs",                       #tensorboard log dir
    "summary_interval"  : 500,                          #num of interval to dump summary
    "val_interval"      : 5000,                         #num of interval to dump summary
    "lr_values"         : "0.00005;0.000005;0.0000005", #lr step values
    "lr_boundaries"     : "150000;300000",              #iters to jump between lr values
    "momentum"          : 0.9,                          #momentum
    "continue_training" : False,                        #resume training from latest checkpoint
    "checkpoint_file"   : "./snaps/pretraindModel/checkpoint.ckpt-1",    #resume from specific ckpt file
    "max_snapshots_keep": 5,                            #max snaps to keep 0 means all
    "output_dir"        : "infer_out"                   #detections images output path

}

config =  edict(default_config)

def printCfg():
    print "CFG: {"
    for (k,v) in config.__dict__.iteritems():
        print "\t\t\"{}\" : {},".format(k , v if type(v) != str else "\"{}\"".format(v))
    print "}"

