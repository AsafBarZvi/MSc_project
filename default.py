import json
from easydict import EasyDict as edict

default_config = {
    "gpu"               : None,                         #gpu device used
    "data_dir"          : ".",                          #data dir containing train.txt test.txt
    "batch_size"        : 20,                           #batch size
    "weight_decay"      : 0.00005,                      #decay on weights
    "bias_decay"        : 0.000005,                     #decay on biases
    "snapdir"           : "./snaps",                    #path to save snapshots
    "epochs"            : 20,                           #number of epoches
    "logdir"            : "logs",                       #tensorboard log dir
    "summary_interval"  : 1000,                         #num of interval to dump summary
    "val_interval"      : 15000,                        #num of interval to run validation
    "lr_values"         : "0.00005;0.000005;0.0000005", #lr step values
    "lr_boundaries"     : "200000;300000",              #iters to jump between lr values
    "momentum"          : 0.9,                          #momentum
    "continue_training" : False,                        #resume training from latest checkpoint
    "checkpoint_file"   : None,                         #resume from specific ckpt file
    "max_snapshots_keep": 15,                           #max snaps to keep 0 means all
    "output_dir"        : "infer_out"                   #detections images output path

}

config =  edict(default_config)

def printCfg():
    print "CFG: {"
    for (k,v) in config.__dict__.iteritems():
        print "\t\t\"{}\" : {},".format(k , v if type(v) != str else "\"{}\"".format(v))
    print "}"


