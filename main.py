import numpy as np
import random
import h5py
import argparse
import os
import glob

from config import Config

def main():
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    model_ckpt = config.model_ckpt
    print(model_ckpt)
    #TODO: created multiple buffer files?
    list_of_files = glob.glob(config.model_path + '/*')
    if not list_of_files and False:
        try:
            # create a buffer file
            with h5py.File("buffer", "w") as f:
                # The number of the last dimension of /pi is config.simulation_num_per_move, not NUM_VOCAB
                # This is because otherwise I/O will take too much time
                # An economical (de)compression process is used
                pi_buffer = f.create_dataset('/pi', shape=(config.buffer_size, config.max_length, config.simulation_num_per_move), )
                s_buffer = f.create_dataset('/s', shape=(config.buffer_size, config.max_length), )
                #z_buffer = f.create_dataset('/z', shape=(config.buffer_size, config.max_length), )
                cur_row = f.create_dataset('/cur_row', shape=(1,), )

                cur_row[0] = 0
                #s_buffer[:,0] = config.start_token
                #s_buffer[:,1:] = config.blank_token
        except Exception:
            pass

    CMD_LIST = ['self', 'opt', 'test']
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--process_ind", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "self":
        import self_play
        return self_play.start(config, args.process_ind)
    elif args.cmd == 'opt':
        import optimize
        return optimize.start(config)
    #TODO: do something for test case
    elif args.cmd == 'test':
        import self_play
        return self_play.start(config, args.process_ind)

if __name__ == '__main__':
    main()
