import os
import numpy as np
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
    
    NUM_EXPERIMENTS=3

    hyp_opt_dir = "hyp_opt_lr_3"
    os.makedirs(hyp_opt_dir,exist_ok=True)
    

    for d in [0.25,0.5,1,2,3,4]:
        os.makedirs(os.path.join(hyp_opt_dir,"d_"+str(d)),exist_ok=True)
        print("testing udr: ",d)

        for i,lr in enumerate(np.logspace(-5,-1,n_lr)):
            print("testing lr: ",lr)
            os.makedirs(os.path.join(hyp_opt_dir,"d_"+str(d),"lr_"+str(lr)),exist_ok=True)

            for experiment in range(NUM_EXPERIMENTS): 

                cmd_str="python main.py --output-dir "+os.path.join(hyp_opt_dir,"d_"+str(d),"lr_"+str(lr),str(experiment))+\
                    " --lr "+str(lr)+\
                    " --timesteps "+str(n_train_steps)+\
                    " --lr-scheduling constant"+\
                    " --use-udr only"+\
                    " --udr-var "+str(d)

                print(i,") ",cmd_str)

                #os.system(cmd_str)

                process_handles.append(Popen([cmd_str], shell=True,
                        stdin=None, stdout=None, stderr=None))
            
                if len(process_handles)%n_proc_parall==0:    
                    for handle in process_handles:
                        handle.wait()
                    process_handles.clear()
    
    for handle in process_handles:
                handle.wait()


    train_env="source"
    fig,ax = plt.subplots()
    
    for curr_dir in os.listdir(hyp_opt_dir):
        log_path=os.path.join(hyp_opt_dir,curr_dir,"train_logs" ,train_env+'_train_log.csv' )
        if os.path.exists(log_path):
            df = pandas.read_csv(log_path)
            ax.plot(df["time/total_timesteps"],df["rollout/ep_rew_mean"],label="lr: "+'{:.2e}'.format(float(curr_dir[3:])))
    
    ax.set_xlabel('timesteps')
    ax.set_ylabel('mean episode rewards')
    ax.set_title("source lr comparison")
    ax.legend(ncol=2) #loc='upper center'  bbox_to_anchor=(0.5, 1.05)
    fig.savefig(os.path.join(hyp_opt_dir,"lr_comparison_curve.svg"))
    

    fig,ax = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()

    source_env_name="CustomHopper-source-v0"
    target_env_name="CustomHopper-target-v0"

    for udr_dir in os.listdir(hyp_opt_dir):
        lrs=[]
        ss_rew=[]
        st_rew=[]
        tt_rew=[]

        #if udr_dir in ["d_0.25"]:
        #    continue

        for lr_dir in os.listdir(os.path.join(hyp_opt_dir,udr_dir)):
            if lr_dir=="lr_0.1":
                continue
            if os.path.isdir(os.path.join(hyp_opt_dir,udr_dir,lr_dir)):
                lrs.append(float(lr_dir[3:]))
                print(udr_dir," ",lr_dir)
                lr_ss_rewards=[]
                lr_st_rewards=[]

                #for exp_dir in os.listdir(os.path.join(hyp_opt_dir,udr_dir,lr_dir)):
                    
                with open(os.path.join(hyp_opt_dir,udr_dir,lr_dir,"test_results.txt")) as res_file: #exp_dir
                    lines=res_file.readlines()
                    if len(lines)==0:
                        model = Model(source_env_name, target_env_name,output_dir="",vision=False)
                        model.load_model(os.path.join(hyp_opt_dir,udr_dir,lr_dir,"checkpoints","source_500000_steps.zip")) #exp_dir
                        ss_curr_rew,ss_curr_std=model.test(test_env_name=source_env_name,n_eval=1)
                        #lines.append("Source-source: "+f"mean_reward={s_mean_rew:.2f} +/- {s_std_rew:.2f}") 
                        st_curr_rew,st_curr_std=model.test(test_env_name=target_env_name,n_eval=1)
                        #lines.append("Target-target: "+f"mean_reward={t_mean_rew:.2f} +/- {t_std_rew:.2f}")
                    
                    else:
                        ss_curr_rew,ss_curr_std=lines[0].split('=')[1].split('+/-')
                        st_curr_rew,st_curr_std=lines[1].split('=')[1].split('+/-')
                    
                    lr_ss_rewards.append(float(ss_curr_rew))
                    lr_st_rewards.append(float(st_curr_rew))

                ss_rew.append(np.mean(lr_ss_rewards))
                st_rew.append(np.mean(lr_st_rewards))
                #tt_curr_rew,tt_curr_std=lines[2].split('=')[1].split('+/-')
                #tt_rew.append(float(tt_curr_rew))
        sort_val=np.array([list(e) for e in zip(lrs,ss_rew,st_rew)])
        sort_val=sort_val[sort_val[:,0].argsort()]
        #sort_val=np.sort([list(e) for e in zip(lrs,ss_rew,st_rew)],axis=0) #,tt_rew
        #print(sort_val)
        ax.plot(sort_val[:,0],sort_val[:,1],label=udr_dir)
        ax2.plot(sort_val[:,0],sort_val[:,2],label=udr_dir)

        ax3.plot(sort_val[:,0],sort_val[:,2]/sort_val[:,1],label=udr_dir)
        #plt.show()
    ax.set_xscale('log')
    ax.set_xlabel('learning rates')
    ax.set_ylabel('source-source mean reward')
    ax.set_title("udr learning rate optimization")
    ax.legend()

    ax2.set_xscale('log')
    ax2.set_xlabel('learning rates')
    ax2.set_ylabel('source-target mean reward')
    ax2.set_title("udr learning rate optimization")
    ax2.legend()

    ax3.set_xscale('log')
    ax3.set_xlabel('learning rates')
    ax3.set_ylabel('source-target ratio')
    ax3.set_title("udr evaluation")
    ax3.legend()
    
    plt.show()