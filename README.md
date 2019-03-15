# Project_RL
Implementation of the Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)) and Maximum A Posteriori
Policy Optimization ([MPO](https://arxiv.org/abs/1806.06920)) Reinforcement Learning Algorithms for continuous 
control on [OpenAI gym](https://github.com/openai/gym) environments.

## Prerequisites
To use the Algorithms will require python3 (>=3.6.5).

* This should install all dependencies and the packages (cloning -> into a directory of choice):
    ```bash
    git clone https://github.com/theogruner/rl_pro_telu
    cd rl_pro_telu
    pip install -e .
    ``` 
 
* Or they can be installed manually

    Installation gym: https://github.com/openai/gym

    Installation quanser: https://git.ias.informatik.tu-darmstadt.de/quanser/clients/tree/master

    Additionally [PyTorch](https://pytorch.org), [Numpy](https://www.scipy.org/scipylib/download.html), 
    [Tensorflow](https://www.tensorflow.org/install), 
    [TensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html) are required.

## Usage Examples
### Note
The algorithms are intended for continuous gym environments !
### DDPG
 * **Usage with provided noises (Ornstein Uhlenbeck, Adaptive Parameter)**

    If you want to use DDPG with our pre-implemented noises
    you can simply do this via the _main_ddpg.py_ file.
        
    * Training on the Qube-v0 (can be replaced with any environment id e.g. Levitation-v0) environment with default hyperparameters and Ornstein-Uhlenbeck noise, 
    saving the model as _furuta_model.pt_ and the log in _furuta_model_
    (saves by default as _ddpg_model.pt_, and log in a automatic generated directory).
    Saving/logging could be disabled with _--no-save_ and _--no-log_
        
        ```bash
        python3 path/to/main_ddpg.py --env Qube-v0 --save_path furuta_model.pt --log_name furuta_log
        ```
    
    * Loading a saved model and evaluating it in the Qube-v0 environment.
     Number of episodes and their length for testing can be adapted with _--eval_episodes_
     and _--eval_ep_length_
    
        ```bash
        python3 path/to/main_ddpg.py --env Qube-v0 --no-train --eval --load furuta_model.pt
        ```
        
    * Training and evaluating are executed sequentially, so if u want to evaluate a just 
    trained model, this will work:
    
        ```bash
        python3 path/to/main_ddpg.py --env Qube-v0 --train -eval --save_path furuta_model.pt --log_name furuta_log 
        ```
        
    * Loading a model and setting the train flag will continue training on the loaded model
    
        ```bash
        python3 path/to/main_ddpg.py --env Qube-v0 --train -eval --save_path furuta_model.pt --log_name furuta_log --load furuta_model.pt
        ```
        
    * Adapting hyperparameters is quite intuitive:
    
        ````bash
        python3 path/to/main_ddpg.py --env Qube-v0 --gamma 0.5 --tau 0.1 --batch_size 1024
        ````

    * For more information, e.g. about adaptable hyperparameters, use:
    
        ```bash
        python3 path/to/main_ddpg.py --help
        ```
    
 * **Usage with self defined noise**
 
   To use a self defined noise you will need to write a script by yourself.
   
   * Make sure the script is the same directory as the _ddpg_ package (only if you didn't install them with [PyPI](https://pypi.org)).
   * The noise should extend the _Noise_ class in _noise.py_ (contain a _reset_ and _iteration_ function) 
   * Following examples cover previous examples functionality
   
        ```python
        import gym    
        import quanser_robots
    
        from ddpg import DDPG
        from ddpg import OrnsteinUhlenbeck
     
        # create environment and noise
        env = gym.make('Qube-v0')
        action_shape = env.action_space.shape[0] 
        noise = OrnsteinUhlenbeck(action_shape)
        
        # setup a DDPG model w.r.t. the environment and self defined noise
        model = DDPG(env, noise, save_path="furuta_model.pt", log_name="furuta_log")
    
        # load a model
        model.load_model("furuta_model.pt")
        #trains the model
        model.train()
        #evaluates the model and returns a meaned reward over all episodes
        mean_reward = model.eval(episodes=100, episode_length=500)
        print(mean_reward)
        
        # always close the environment when finished 
        env.cose()
        ``` 
    * Setting Hyperparameters would look something like this:
 
       ```python
       model = DDPG(env, noise, gamma=0.5, tau=0.1, learning_rate=1e-2)
       ```
 * **Using a model as a controller**
    
    If you want to use your a model as a simple controller just call the model
    passing an observation:
    
    ```python
    ctrl = DDPG(env, noise)
    ctrl.load('furuta_model.pt')

    while not done:
       env.render()
       act = ctrl(obs)
       obs, rwd, done, info = env.step(act)
    
    # always close the environment when finished
    env.close()
    ```
### MPO

Using MPO is analogous to ddpg

* Use _main_mpo.py_ instead of _main_ddpg.py_
* For information on the parameters you can set: _python3 main_mpo.py --help_
* Due to no noise needed writing an own script is not necessary but for completeness 
a little code snippet as example:

    ```python
    import gym
    import quanser_robots
    from mpo import MPO

    env = gym.make('Qube-v0')
    model = MPO(env, save_path="furuta_model.pt", log_name="furuta_log")
    # continues like DDPG example ...
    ```

### Logging
By default logging is enabled and safes the logs in the _runs/_ directory.
Name of the logs can be set by the *log_name* parameter (*--log_name* argument).
Inspecting them works with:

```bash
tensorboard --logdir=*/PATH/TO/runs*
```
This starts a local server, which can be accessed in the browser.
Connecting to the server should result in something like this:

![tensorboar](data/tensorboard.png)

## Open Source Infos
### Contributing
Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

### Authors
* **theogruner**      - [theogruner](https://github.com/theogruner)
* **DariusSchneider** - [DariusSchneider](https://github.com/DariusSchneider)

See also the list of [contributors](https://github.com/theogruner/Project_RL/contributors) who participated in this project.

### License
This project is licensed under the GNU GPL3 License - see the [LICENSE](LICENSE) file for details
