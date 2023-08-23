import os
from model import Model
import gym
from stable_baselines3.common.vec_env import VecVideoRecorder,DummyVecEnv
import mujoco_py
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #os.system("Xvfb :1 -screen 0 1024x768x24 &")
    #os.environ['DISPLAY'] = ':1'

    checkpoint="prova_vec_env_udr_2/checkpoints/SAC_s_lr_0.0003_steps_1000000" #source_2000000_steps.zip
    source_env_name="CustomHopper-source-v0"
    target_env_name="CustomHopper-target-v0"
    n_test_eps=50
    use_vision=False

    model = Model(source_env_name, target_env_name,output_dir="",vision=use_vision)
    model.load_model(checkpoint)

    print("Source: ")
    s_mean_rew,s_std_rew=model.test(test_env_name=source_env_name,n_eval=n_test_eps)
    print("Target: ")
    t_mean_rew,t_std_rew=model.test(test_env_name=target_env_name,n_eval=n_test_eps)
    rend_env=gym.make(source_env_name) #DummyVecEnv([lambda:

    
    rend_env=VecVideoRecorder(rend_env,
        video_folder="./videos1",
        record_video_trigger=lambda step: step == 0,
        video_length=1000,
        name_prefix="")
    
    obs = rend_env.reset()
    
    for i in range(1_000):
        
        action, _states = model.model.predict(obs, deterministic=True)
        obs, rewards, dones, info = rend_env.step(action)

        """
        Attenzione: render() equivale a mode="human", accede a glfw, per cui ha bisogno di:
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
        mentre rgb_array ha bisogno che LD_PRELOAD non sia settata, fare unset LD_PRELOAD.
        Controllare anche che openGL usi la stessa gpu di render (stampa qui "using device:0", su terminale glxinfo -B)
        eventualmente settare D3D12, ad esempio:
        export MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA GeForce RTX 3050 Ti Laptop GPU"
        """

        #img=rend_env.render("rgb_array",width=224, height=224) #"rgb_array",
        #rend_env.render()

        #img=rend_env.env.sim.render(width=224, height=224)
        
        #img = rend_env.sim.render(width=224, height=224) #"rgb_array",
        
        #plt.imshow(img)
        #plt.show()
    
    rend_env.close()


        
