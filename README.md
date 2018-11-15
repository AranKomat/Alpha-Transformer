# Alpha-Transformer
Alpha Zero equipped with Transformer with various novel techniques for speedup in tree search

About
=====

Alpha Transformer, or AT, is an algorithm and implementation of Alpha Zero based on Transformer with various novel speedup techniques applicable to tree search in general as well as Transformer on tree. We benchmarked its performance on various single-agent tasks by utilizing ranked reward as in [Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization](https://arxiv.org/abs/1807.01672). 


It can solve sequential decision problems effectively with minimal computation resources. Our implementation is for single-agent tasks only; however, one can easily modify our implementation to convert it to two-agent version as in the original Alpha Zero. For this and more detailed comments on our implementation of Alpha Zero, please refer to [reversi-alpha-zero](https://github.com/mokemokechicken/reversi-alpha-zero) by [mokemokechiken](https://github.com/mokemokechicken/), which inspired our implementation of Alpha Zero. 

Let D, L and N to be the hidden dimension, episode length and the number of simulations per move and  , respectively. 

The contribution of this project includes:

1. ... performance in the benchmark with a single GPU ...

2. To devise a method in which hidden states (caches) are stored in each already evaluated node of tree, thus avoiding redundancy of inference, and organized in a way such that the GPU memory consumption and CPU-GPU transfer are minimized. 

The sequential decision problem of our interest can be represented as a string of actions as in the setting of Alpha Zero. Naively using a CNN or Transformer to evaluate on the entire string at each timestep costs quadratically to the episode length. If we employ either RNN or Transformer with caches stored in each already evaluated node, the cost can be reduced to linear w.r.t. the episode length. 

CPU-GPU transfer in both directions becomes a bottleneck due to the nature of our algorithm. This was circumvented by the use of PyTorch and a sophisticated management of stored caches. In the case of Tensorflow, the standard procedure transfers the input to GPU and then back to CPU. In the case of PyTorch, it straightforward to always keep the caches on GPU, avoiding the unnecessary transfer. Furthermore, what is transferred between the server and the clients is not the caches themselves but the string representation of each state. This allows the caches to be always stored in the server and hence avoids a large transfer throughput. Since each client does not store the caches overlapped with the ones stored by other clients, it also reduces the GPU memory consumption. Each stored cache is segmented timestep-wise, and each overlapping segmented cache is discarded for further saving.  

3. To devise a method that significantly reduces the computational cost and GPU memory consumption in matrix multiplications in self-attention upon inference on tree with Transformer. 

The method exploits the tendency that there is a significant overlap between the cache of each tree node, and the overlapped caches can be replaced with a single cache. To achieve this in inference, a dynamic modification of mask of attention is devised. Note that the method does not reduce the cost due to evauation of Q, V and K from the previous hidden state with linear layer, which is not a bottleneck in our setting.   

This reduces the memory consumption of a single root state evaluation, consisting of evaluating hundreds of the child nodes, from O(DLN) to O((L+D)N). The difference is so large that, even on a GPU with large enough memory to afford O(DLN), the total speed differs by the factor of five on a toy problem. 

4. To offer an implementation with various miscellaneous hacks to reduce the I/O and CPU computation cost.  

There are various hacks used in our implementation to reduce the I/O and CPU computation cost, which also become a bottleneck in achieving the speedup we gained. For example, the size of `/buffer` quickly grows to the order of GB. The maximum size of `/buffer` can be calculated as 8 x `config.buffer_size` x LN bytes. This does not depend on the number of possible actions but instead on a usually smaller N, since we store the visit counts not as a distribution. Also, the use of h5py is essential to the speed. Reading the rows of the previous visit counts becomes a bottleneck, but this was circumvented by reading the small chuncks iteratively instead of reading a large chunck at once. The use of PyTorch is also essential for quickly updating the model weights. With Tensorflow, one has to destroy the existing graph before updating weights, which makes the process significantly longer. The calculation of Dirichlet distribution was a CPU bottleneck, which was resolved by our approximation precise enough for practical use. 


Some possible future directions include to combine the methods introduced in the following papers:

1. [How to Combine Tree-Search Methods in Reinforcement Learning](https://arxiv.org/abs/1809.01843) for better tree search
2. [Learning to Search with MCTSnets](https://arxiv.org/abs/1802.04697) for meta-MCTS
3. [Dual Policy Iteration](https://arxiv.org/abs/1805.10755) for continuous settings
4. [Probabilistic Planning with Sequential Monte Carlo](https://openreview.net/forum?id=ByetGn0cYX) for planning

It is also crucial to compare our algorithm with competitors such as [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857) and similar other recent works.  


We expect our algorithm to replace REINFORCE and PPO in the setting of discrete optimization. Though REINFORCE is the default choice for RL on discrete objects in, for example, NLP and architecture optimization, we believe our algorithm can be the new REINFORCE without overfitting, provided that we can simplify the usage of our algorithm, an important future work of our project.   

For more details, please refer to our paper. 
 

Results
-----------



Environment
-----------

* Python 3.6 
  * It works with Python 3.5 by replacing `from _asyncio import Future` with `from asyncio import Future`, but I cannot guarantee that it will reach to the same speed. 
* PyTorch: >= 0.4.0
  
How to use
==========

Basic Usage
------------
To run self-play workers, enter the following command in a terminal (or possibly multiple separate terminals).  
```bash
python3 main.py self
```
Soon after that, enter the following command to begin the training in a single terminal separate from the ones used above: 
```bash
python3 main.py opt 
```
To reset the training, you need to either remove or relocate not only all the ckpt files in the corresponding folder but also `/buffer`. 

Modules
-------
`/arch.py` contains all the architectures. 


Hyperparameters
--------------
The general hyperparameters are stored at `config.py`. The default hyperparameters allow the optimal performance in a typical single GPU and a single CPU. To optimize for your environment, adjust the hyperparameters as follows. `config.batch_size` should be larger, but it should be typically somewhere from 256 to 2048. if the average episode length is long, it could be smaller. `config.batch_size` is roughly proportional to the time required for an iteration, so it should be set so that a single iteration takes not too long for your convenience. Note that batch size here refers to the number of episodes per iteration rather than the number of timesteps per iteration. Then, you can also adjust `config.prediction_queue_size`, `config.parallel_search_num` and `config.multi_process_num` to optimize the speed. If the time of processing the number of episodes equal to `config.batch_size` is much greater than the time interval of each gradient update, you should increase `config.training_sleep_time` for better balance. The former should not be less than the latter. If your GPU is not fully utilized and the hyperparameter tuning does not improve the situation, it is advisable to add another set of self-play workers by entering:

```bash
python3 main.py self
```


Tips and Memo
====

Caveats
----------
Beware that older ckpt files are automatically deleted, and the buffer file cannot be checkpointed.   

The size of `/buffer` rather quickly grows to the order of GB. The maximum size can be calculated as 8 x `config.buffer_size` x `config.max_length` x `config.simulation_num_per_move` bytes.

We used h5py and various hacks to reduce I/O computation.

Note that our random seed is technically not reproducible, since We needed to use `np.seed(None)`for multiprocessing.

GPU Memory
----------



Training Speed
------
The following is obsolated info
------------


* Hardware: g3.4xlarge (1 M60, 8GB GPU memory, 16 vCPUs, which in total are probably equivalent to a half of Xeon E5-2686 v4) 
* 1 iteration with LSTM and batch size 64: about 7 sec
  * GPU usage rate was 40% on average. 
  * 1 gradient update: about 0.4 sec
  * I/O operations per iteration: about 0.1 sec
* It takes a much longer time for a single iteration than in theory. This is presumably because NMG takes a considerable amount of CPU-to-GPU data transfer due to the cache used in inference. The residual layer-norm 8-layered LSTM resolved this issue to some extent. In fact, the LSTM requires 12 times less amount of cache in this particular case, and for sequences with an arbitrary length, the amount of cache required for inference is constant. Due to the lack of computation resources, we defer experiments with Transformer and focus on using LSTM for a while. 
