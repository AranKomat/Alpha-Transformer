# Alpha-Transformer
Alpha Zero equipped Transformer with various techniques for speedup

About
=====

Alpha Transformer, or AT, is an algorithm and implementation of Alpha Zero based on Transformer with various speedup techniques to solve sequential decision problems effectively with minimal computation resources. For the details, please refer to the paper. 

Though AlphaZero required 5000 TPUs for game generation, the FLOPS for a single move in my case is smaller than that of AZ by the order of 1000. 

Hidden states are stored in each already evaluated node of tree, thus avoiding redundancy of inference. 

This implementation was inspired from Alpha Zero at [reversi-alpha-zero](https://github.com/mokemokechicken/reversi-alpha-zero) by [mokemokechiken](https://github.com/mokemokechicken/). Transformer part is based on [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). The part responsible for text generation task proper is based on [LeakGAN](https://github.com/CR-Gjx/LeakGAN/). LSTM part is based on [Google-Neural-Machine-Translation-GNMT](https://github.com/shawnxu1318/Google-Neural-Machine-Translation-GNMT). One can refer to these links for details of my implementation in respective components. 

An idea regarding symbolic board game as a compact version of AlphaZero was presented [here](https://github.com/mokemokechicken/reversi-alpha-zero/issues/50). 

Results
-----------



Environment
-----------

* Python 3.6 
  * It works with Python 3.5 by replacing `from _asyncio import Future` with `from asyncio import Future`, but I cannot guarantee that it will reach to the same speed. 
* PyTorch: >= 0.4.0
  
How to use
==========
Modules
-------
`Main.py` generates texts by calling `self_player.py`, store them in `buffer`, trains the model and periodically outputs generated texts translated to natural language `save/sample.txt` as well as BLEU scores in `save/BLEU.txt`. 

The folder `tensor2tensor` and the file `Model.py` contain all the necessary ingredients of Transformer, and they are currently not used. 

Hyperparameters
--------------
There are three locations where you can find hyperparameters to be defined: `Main.py` and `config.py`. The ones defined in `Main.py` are mainly about text generation task proper and rarely used in other files. The ones defined in `config.py` are mainly about AlphaZero. 

Basic Usage
------------
Running Main.py performs all the jobs as explained above: 

```bash
python3 Main.py
```
Tips and Memo
====

Caveats
----------
The size of `buffer` rather quickly grows, and its total size grows to about 5GB. I used h5py and various hacks to reduce I/O computation.

Note that my random seed is technically not reproducible, since I needed to use `np.seed(None)`for multiprocessing.

GPU Memory
----------
The default hyperparameters allow the optimal performance in my environment. To optimize for your environment, adjust the hyperparameters as follows. `batch_size` should be larger, but it should be somewhere from 32 to 2048. `batch_size` is proportional to the time required for an iteration, so it should be set so that a single iteration takes not too long for your convenience. Note that batch size here refers to the number of episodes per iteration rather than the number of timesteps per iteration. Then, you can proportionally scale `prediction_queue_size`, `parallel_search_num` and `multi_process_num` to optimize the speed. If your computation resources allow, you are encouraged to use a larger architecture. 

Training Speed
------
* Hardware: g3.4xlarge (1 M60, 8GB GPU memory, 16 vCPUs, which in total are probably equivalent to a half of Xeon E5-2686 v4) 
* 1 iteration with LSTM and batch size 64: about 7 sec
  * GPU usage rate was 40% on average. 
  * 1 gradient update: about 0.4 sec
  * I/O operations per iteration: about 0.1 sec
* It takes a much longer time for a single iteration than in theory. This is presumably because NMG takes a considerable amount of CPU-to-GPU data transfer due to the cache used in inference. The residual layer-norm 8-layered LSTM resolved this issue to some extent. In fact, the LSTM requires 12 times less amount of cache in this particular case, and for sequences with an arbitrary length, the amount of cache required for inference is constant. Due to the lack of computation resources, I defer experiments with Transformer and focus on using LSTM for a while. 
