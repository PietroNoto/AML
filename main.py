from os.path import exists
import argparse
import os
from model import Model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",         type=str, help='Path where the model is saved', default="output")
    parser.add_argument("--test-eps",           type=int, help='Number of test episodes', default=50)
    parser.add_argument("--eval-target-target", action='store_true', help='Evaluate Target-Target loss')
    parser.add_argument("--lr",                 type=float, help='Starting learning rate', default=1e-3)
    parser.add_argument("--lr-scheduling",      type=str, help='Learning rate scheduling', default="constant", choices=["constant","linear"]) #aggiungere "cosine"
    parser.add_argument("--timesteps",          type=int, help='Training timesteps', default=100_000)
    parser.add_argument("--buffer-size",        type=int, help='Buffer size', default=1_000_000)
    parser.add_argument("--use-vec-env",        action='store_true', help='Use vectorized environments')
    parser.add_argument("--use-udr",            action='store_true', help='Use Uniform Domain Randomization')
    parser.add_argument("--udr-type",           type=str, help = 'Type of UDR technique', default = "infinite", choices = ["finite", "infinite"])
    parser.add_argument("--udr-lbound",         type=int, help = 'Finite UDR - Lower bound', default = 1)
    parser.add_argument("--udr-ubound",         type=int, help = 'Finite UDR - Upper bound', default = 3)
    parser.add_argument("--n-distr",            type=int, help='Finite UDR - Number of distributions', default = 3)
    parser.add_argument("--udr-range",          type=float, help = 'Infinite UDR - Relative range', default = 2)
    parser.add_argument("--source-env",         type=str, help='Source environment name', default="CustomHopper-source-v0")
    parser.add_argument("--target-env",         type=str, help='Target environment name', default="CustomHopper-target-v0")
    parser.add_argument("--checkpoint",         type=str, help='Path of a checkpoint file', default=None)
    parser.add_argument("--use-vision",         action='store_true', help='Change observation space to pixels')
    parser.add_argument("--width",              type=int, help='Width of rendered image', default=224)
    parser.add_argument("--height",             type=int, help='Height of renedered image', default=224)
    parser.add_argument("--grayscale",          action='store_true', help='Change pixels from rgb to grayscale')
    parser.add_argument("--use-pose-est",       action='store_true', help='Use a custom pose estimation network')
    parser.add_argument("--mmpose-config",      type=str, help='Path of MMPose config', default="")
    parser.add_argument("--mmpose-checkpoint",  type=str, help='Path of MMPose checkpoint', default="")

    args=parser.parse_args()
    
    n_test_eps = args.test_eps
    eval_tt = args.eval_target_target
    lr = args.lr
    lr_sched = args.lr_scheduling
    n_timesteps = args.timesteps
    buf_size = args.buffer_size
    use_vec_env = args.use_vec_env
    use_udr=args.use_udr
    udr_type = args.udr_type
    udr_lb = args.udr_lbound
    udr_ub = args.udr_ubound
    n_distr = args.n_distr
    udr_range = args.udr_range
    source_env_name = args.source_env
    target_env_name = args.target_env
    checkpoint = args.checkpoint
    use_vision = args.use_vision
    width = args.width
    height = args.height
    grayscale = args.grayscale
    use_pose_est = args.use_pose_est
    pose_config = args.mmpose_config
    pose_checkpoint = args.mmpose_checkpoint

    #args:  train->plots: output-dir/  n_timesteps  lr  batch_size  checkpoint_file  policy [MLP, CNN]? <- argomenti
    #                       checkpoints/
    #                           checkpoint_1.pt ...   #serve callback, per allenamenti lunghi
    #                       train_logs/progress.csv    #rinomina, per source e target
    #                       plots/                    #distinguere source e plot anche qui
    #                            source_train_plot.svg
    #                            target_train_plot.svg
    #       test -> spostare in un altro script che carica modello e testa

    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'train_logs')

    #se la cartella é già esistente si blocca, perderemmo i file di log, altrimenti crea la struttura
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(logs_dir)
    elif len(os.listdir(output_dir)) > 0:
        raise FileExistsError('Output dir contains files')
    else:
        os.mkdir(checkpoints_dir)

    #stampa file con gli iperparametri
    with open(os.path.join(output_dir,'params.txt'), 'w') as param_file:
        param_str="lr: "+str(lr)+\
            f"\nscheduling: {lr_sched}"+\
            f"\ntotal steps: {str(n_timesteps)}"+\
            f"\nsource env: {source_env_name}"+\
            f"\ntarget env: {target_env_name}"+\
            f"\neval target-target: {str(eval_tt)}"+\
            f"\nbuffer size: {str(args.buffer_size)}"+\
            f"\ntest episodes: {str(n_test_eps)}"+\
            f"\ncheckpoint path: {str(checkpoint)}"
        if use_vec_env:
            param_str+=f"\n---use vec env: {str(use_vec_env)}"
        if use_udr:
            param_str+=f"\n---use udr: {str(use_udr)}"+\
                f"\nudr type: {udr_type}"
            if udr_type == "finite":
                param_str+=f"\nudr lower bound: {str(udr_lb)}"+\
                f"\nudr upper bound: {str(udr_ub)}"+\
                f"\nudr distr n: {str(n_distr)}"
            if udr_type == "infinite":
                param_str+=f"\nudr range: {str(udr_range)}"
        if use_vision:
            param_str+=f"\n---use vision: {str(use_vision)}"+\
            f"\nimage width: {str(width)}"+\
            f"\nimage height: {str(height)}"+\
            f"\ngrayscale: {str(grayscale)}"
        if use_pose_est:
            param_str+=f"\n---use pose estimation: {str(use_pose_est)}"+\
            f"\npose config file: {str(pose_config)}"+\
            f"\npose checkpoint: {str(pose_checkpoint)}"
        param_file.write(param_str)

    #file con i risultati dell'esperimento
    with open(os.path.join(output_dir,'test_results.txt'), 'w') as test_fp:
        model_kwargs = dict(output_dir=output_dir,
                      use_vec_env=use_vec_env,
                      use_udr=use_udr,
                      udr_type=udr_type,
                      udr_lb=udr_lb,
                      udr_ub=udr_ub,
                      udr_ndistr=n_distr,
                      udr_range=udr_range,
                      use_vision=use_vision,
                      width=width,
                      height=height,
                      grayscale=grayscale,
                      pose_est=use_pose_est,
                      pose_config=pose_config,
                      pose_checkpoint=pose_checkpoint)
        
        model = Model(train_env_name=source_env_name,
                      test_env_name=target_env_name,
                      **model_kwargs)
        if args.checkpoint != None:
            model.load_model(args.checkpoint)
        print("Source-target:")
        model.train(timesteps=n_timesteps, learning_rate = lr,lr_schedule=args.lr_scheduling,buffer_size=args.buffer_size)
        model.plot_results()
        st_mean_rew, st_std_rew = model.test(n_eval=n_test_eps)
        test_fp.write("Source-target: " + f"mean_reward={st_mean_rew:.2f} +/- {st_std_rew:.2f}")

        print("Source-source:")
        model.change_env("test", source_env_name)
        ss_mean_rew, ss_std_rew = model.test(n_test_eps)
        test_fp.write("\nSource-source: " + f"mean_reward={ss_mean_rew:.2f} +/- {ss_std_rew:.2f}")

        if (eval_tt):
            print("Target-target:")
            model = Model(train_env_name=target_env_name,
                        test_env_name=target_env_name,
                        **model_kwargs)
            if args.checkpoint != None:
                model.load_model(args.checkpoint.replace("source","target"))
            model.train(timesteps=n_timesteps, learning_rate = lr,lr_schedule=args.lr_scheduling,buffer_size=args.buffer_size)
            model.plot_results()
            tt_mean_rew, tt_std_rew = model.test(n_eval=n_test_eps)
            test_fp.write("\nTarget-target: "+f"mean_reward={tt_mean_rew:.2f} +/- {tt_std_rew:.2f}")