
import os
import numpy as np

#from concurrent.futures.thread import ThreadPoolExecutor
from subprocess import Popen
import matplotlib.pyplot as plt
import pandas

if __name__=='__main__':
    """
    does hyperparameter optimization, trying different:
    lr
    buffer size
    - batch size; non dovrebbe influire, in verità io ti dico
    lr scheduling
    learning-starts?
    target entropy

    test_eps
    seeds
    best?
    tot-points

    1 lr [0.0001 -> 0.01] [5e-4, 1e-2]
    2 buffer size [1, 100, 1000, 10_000, 100_000, 1_000_000]
    3 random grid
    """
    n_train_steps = 100_000
    n_lr = 10
    n_proc_parall = 3
    process_handles=[]

    hyp_opt_dir = "hyp_opt_lr_3"
    os.makedirs(hyp_opt_dir,exist_ok=True)
    
    for i,lr in enumerate(np.logspace(-5, -1, n_lr)):
        print("testing lr: ",lr)
        
        cmd_str = "python3 main.py --output-dir " + os.path.join(hyp_opt_dir,"lr_"+str(lr))+\
            " --lr "+str(lr) +\
            " --timesteps " + str(n_train_steps)
        print(i,") ",cmd_str)

        os.system(cmd_str)

        process_handles.append(Popen([cmd_str], shell=True,
                stdin=None, stdout=None, stderr=None))
        
        if (i+1) % n_proc_parall == 0:    
            for handle in process_handles:
                handle.wait()
            process_handles.clear()
    for handle in process_handles:
                handle.wait()
    
    train_env = "source"
    fig,ax = plt.subplots()
    for curr_dir in os.listdir(hyp_opt_dir):
        log_path = os.path.join(hyp_opt_dir,curr_dir,"train_logs" ,train_env+'_train_log.csv' )
        if os.path.exists(log_path):
            df = pandas.read_csv(log_path)
            ax.plot(df["time/total_timesteps"],df["rollout/ep_rew_mean"],label="lr: "+'{:.2e}'.format(float(curr_dir[3:])))
    ax.set_xlabel('timesteps')
    ax.set_ylabel('mean episode rewards')
    ax.set_title("source lr comparison")
    ax.legend(ncol=2) #loc='upper center'  bbox_to_anchor=(0.5, 1.05)
    fig.savefig(os.path.join(hyp_opt_dir,"lr_comparison_curve.svg"))





