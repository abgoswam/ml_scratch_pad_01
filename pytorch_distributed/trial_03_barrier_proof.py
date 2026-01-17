"""
Demonstrating that init_process_group is a synchronization barrier.
Process 0 will sleep before calling init_process_group.
If it's a barrier, Process 1 will WAIT even though it arrives first.
"""
#!/usr/bin/env python
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """After init_process_group - both should print at roughly the same time"""
    print(f"[{time.strftime('%H:%M:%S')}] Rank {rank}: Now inside run() - AFTER the barrier")


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    
    # Rank 0 will sleep for 3 seconds BEFORE init_process_group
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] Rank {rank}: Sleeping for 3 seconds BEFORE init_process_group...")
        time.sleep(10)
        print(f"[{time.strftime('%H:%M:%S')}] Rank {rank}: Woke up, now calling init_process_group...")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Rank {rank}: Calling init_process_group immediately (no sleep)...")
    
    # THIS IS THE BARRIER - both processes must arrive here before either proceeds
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    print(f"[{time.strftime('%H:%M:%S')}] Rank {rank}: init_process_group COMPLETED - passed the barrier!")
    
    fn(rank, size)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    
    print(f"[{time.strftime('%H:%M:%S')}] Main: Starting {world_size} processes...")
    
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    print(f"[{time.strftime('%H:%M:%S')}] Main: All processes finished")
