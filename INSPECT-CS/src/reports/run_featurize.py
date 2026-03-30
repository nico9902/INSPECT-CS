import hydra
import re
import logging
import torch
import h5py
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def save_hdf5_features(features, impression_id, hdf5_file):
    """
    Save features to HDF5 file.

    Args:
        features (torch.Tensor): The features to save.
        impression_id (str): The impression ID for the sample.
        hdf5_file (h5py.File): HDF5 file object.
    """
    try:
        hdf5_file.create_dataset(impression_id, data=features, dtype='float32')
        logging.info(f"Added impression {impression_id} to HDF5 file with shape {features.shape}")
    except Exception as e:
        logging.error(f"[ERROR] {impression_id}: {str(e)}")

def custom_sentence_splitter(text):
    """
    Custom function to split medical text into sentences while preserving numbered findings.
    """
    # Rule 1: Ensure that periods within numbers (e.g., "1. PROBABLE") are handled correctly
    text = re.sub(r"(\d+)\.\s", r"\1###", text)  # Temporarily replace numbered list dots

    # Rule 2: Split using standard sentence-ending punctuation
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)

    # Rule 3: Restore numbered findings
    sentences = [s.replace("###", ". ") for s in sentences]

    return [s.strip() for s in sentences if s.strip()]  # Remove empty entries

def load_text_embeddings(impression, cfg, device="cpu"):
    """
    Encodes a list of medical impressions into numerical embeddings using a transformer model.
    """
    # Load tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.reports_encoder)
    model = AutoModel.from_pretrained(cfg.model.reports_encoder).to(device)

    def process_impression(impression):
        # Split the impression into sentences
        sentences = custom_sentence_splitter(impression)

        # Tokenize all sentences in a batch
        encoded_text = tokenizer(
            sentences,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Get token embeddings for all sentences in a batch
        with torch.no_grad():
            hidden_states = model(**encoded_text, output_hidden_states=True).hidden_states[-1]

        return hidden_states.cpu().detach().numpy()  # Convert to NumPy

    # Process each impression and get its embeddings
    features = process_impression(impression)
    return features

@hydra.main(config_path="./configs", config_name="classify")
def main(cfg):
    import os
    import torch
    import pandas as pd

    # Load data
    try:
        data = pd.read_csv("/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/data/reports/impressions_20250611.tsv", sep='\t')
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        return

    # Determine the range of indices for this job
    total_samples = len(data)
    job_id = int(os.getenv("JOB_ID", 0))
    num_jobs = int(os.getenv("NUM_JOBS", 1))

    samples_per_job = total_samples // num_jobs
    start_idx = job_id * samples_per_job
    end_idx = start_idx + samples_per_job if job_id < num_jobs - 1 else total_samples

    logging.info(f"Job {job_id} processing samples {start_idx} to {end_idx}")

    # Subset the dataset for this job
    data_subset = data.iloc[start_idx:end_idx]

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Starting feature extraction...", flush=True)
    output_path = f"/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/datasets/inspect2/features_job_{os.getenv('JOB_ID', 0)}.hdf5"
    with h5py.File(output_path, 'w') as hdf5_file:
        for _, row in data_subset.iterrows():
            impression_id = row['impression_id']
            impression = row['impressions']
            logging.info(f"Processing impression {impression_id}")
            try:
                features = load_text_embeddings(impression, cfg, device)
                logging.info(f"Features shape: {features.shape}")
                save_hdf5_features(features, impression_id, hdf5_file)
            except Exception as e:
                logging.error(f"Failed to process impression {impression_id}: {str(e)}")

    logging.info("Feature extraction and saving complete.")

if __name__ == "__main__":
    main()