"""Main pretraining script."""
import hydra
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import os
import numpy as np
import threading
import time
from torch.cuda.amp import autocast
import h5py

def save_hdf5_features(features, impression_id, hdf5_file):
    """
    Save features to HDF5 file.

    Args:
        features (torch.Tensor): The features to save.
        impression_id (str): The impression ID for the sample.
        hdf5_file (str): Directory to save the features.
    """
    try:
        hdf5_file.create_dataset(impression_id, data=features, dtype='float32')
        print(f"Added impression {impression_id} to HDF5 file with shape {features.shape}", flush=True)
    except Exception as e:
        print("[ERROR]", impression_id, str(e))

def keep_gpu_busy(model, dummy_input, device, sleep_time=0.05, duration=None):
    """
    Keeps the GPU busy by running dummy computations in a loop.

    Args:
        model (torch.nn.Module): The model to use for dummy computations.
        dummy_input (torch.Tensor): The dummy input tensor.
        sleep_time (float): Time to sleep between computations (in seconds).
        duration (float, optional): Total duration to run the loop (in seconds). If None, runs indefinitely.
    """
    print(f"Starting GPU occupation thread on {device}")
    model.eval().to(device)
    dummy_input = dummy_input.to(device)

    start_time = time.time()
    while duration is None or (time.time() - start_time < duration):
        with torch.no_grad():
            _ = model(dummy_input)  # Perform dummy computation
        time.sleep(sleep_time)      # Sleep to avoid overloading the GPU

    print(f"GPU occupation thread on {device} finished.")

class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x

class Prefetcher:
    """Prefetch data to GPU while the GPU is busy."""
    def __init__(self, dataset, device, max_prefetch=2):
        self.dataset = dataset
        self.device = device
        self.queue = Queue(maxsize=max_prefetch)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stop_signal = False

    def producer(self):
        """Load data to the queue."""
        for sample in self.dataset:
            if self.stop_signal:
                break
            x, y, impression_id = sample
            x = x.to(self.device, non_blocking=True)
            self.queue.put((x, y, impression_id))
        self.queue.put(None)  # Signal end of data

    def start(self):
        """Start the producer thread."""
        self.executor.submit(self.producer)

    def get_next(self):
        """Get the next prefetched batch."""
        return self.queue.get()

    def stop(self):
        """Stop the producer."""
        self.stop_signal = True
        self.executor.shutdown(wait=True)

@hydra.main(config_path="./radfusion3/configs", config_name="extract")
def run(config):
    # Deferred imports for faster tab completion
    import timm
    from radfusion3 import builder

    # Create the output directory if it doesn't exist
    features_dir = config.dataset.output_dir
    if not os.path.isdir(features_dir):
        os.makedirs(features_dir)

    # Load the model
    model = timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)
    model.head.fc = Identity()
    checkpoint = torch.load(
        "/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/experiments/model_dir/rsna_pe/resnetv2_ct.ckpt",
    )
    ckpt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith('classifier')} # Filter out classifier keys if they exist and you don't need them 
    msg = model.load_state_dict(ckpt, strict=False)
    print("=" * 80)
    print(msg)
    print("=" * 80)
    
    # Load the dataset
    transform = builder.build_transformation(config, config.test_split)
    dataset2D = builder.build_dataset(config)
    dataset = dataset2D(config, split=config.test_split, transform=transform)

    # Determine the range of indices for this job
    total_samples = len(dataset)
    job_id = int(os.getenv("JOB_ID", 0))  # Unique ID for this job (set via environment variable)
    num_jobs = int(os.getenv("NUM_JOBS", 1))  # Total number of jobs (set via environment variable)

    # Split the dataset into chunks
    samples_per_job = total_samples // num_jobs
    start_idx = job_id * samples_per_job
    end_idx = start_idx + samples_per_job if job_id < num_jobs - 1 else total_samples

    print(f"Job {job_id} processing samples {start_idx} to {end_idx}", flush=True)

    # Subset the dataset for this job
    dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model to device
    model.to(device)
    model.eval()

    # Start the GPU occupation thread
    dummy_input = torch.randn(4, 3, 224, 224)  # Adjust the size as needed
    gpu_thread = threading.Thread(
        target=keep_gpu_busy, 
        args=(model, dummy_input, device),  
        daemon=True
    )
    gpu_thread.start()

    # # Initialize the prefetcher
    # prefetcher = Prefetcher(dataset, device)
    # prefetcher.start()

    # Initialize a thread pool executor for saving features
    save_executor = ThreadPoolExecutor(max_workers=4)

    # Iterate over the dataset
    print("Starting feature extraction...", flush=True)
    output_path = f"/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/datasets/inspect2/features_job_{os.getenv('JOB_ID', 0)}.hdf5"
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        for i in range(len(dataset)):
            # sample = prefetcher.get_next()
            # if sample is None:  # End of data
            #     break

            x, y, impression_id = dataset[i]
            if impression_id == "PE452890e":
                print("Skipping PE452890e", flush=True)
                continue
            x = x.to(device, non_blocking=True)

            batch_size = 128
            num_samples = x.size(0)

            features = []

            with torch.no_grad():  # se stai facendo solo inference
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    x_batch = x[start_idx:end_idx]  # slice batch
                    x_batch = x_batch.to(device, non_blocking=True)

                    output = model(x_batch)
                    features.append(output.cpu().detach())  # scarica da GPU per risparmiare memoria

            # Concateni tutti gli output finali (opzionale)
            final_output = torch.cat(features, dim=0)
            # Convert to numpy array
            final_output = final_output.numpy()

            save_executor.submit(save_hdf5_features, final_output, impression_id, hdf5_file) 

            # print(f"Processing {impression_id}", flush=True)

            # Get the features
            # Use torch.no_grad() to save memory and computations
            # with torch.no_grad():
            #     features = model(x)

            # Save features as individual files if needed  
            # save_executor.submit(save_hdf5_features, features.cpu().detach().numpy(), impression_id, hdf5_file) 
        
        # Stop the prefetcher
        # prefetcher.stop()

        # Shutdown the executor after all tasks are submitted
        save_executor.shutdown(wait=True)
        print("Feature extraction and saving complete.")

if __name__ == "__main__":
    run()