import hydra

@hydra.main(version_base=None, config_path="./configs", config_name="classify")
def main(cfg):
    import os
    import torch
    import numpy as np
    import pytorch_lightning as pl
    from lightning_model import PEModel
    from datamodule import PEDataModule
    import utils_general
    import pickle

    utils_general.seed_all(cfg.seed)
    task = cfg.task
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Task: {task}")
    print(f"Experiment name: {cfg.exp_name}")

    # Initialize the DataModule
    data_module = PEDataModule(cfg, task, device)

    # Initialize the model
    model = PEModel(cfg, task, device, cfg.exp_name)

    # Callbacks
    save_dir = os.path.join("outputs", cfg.exp_name) 

    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory created: {save_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")
    
    if not os.path.exists(save_dir):
        print(f"Directory was not created: {save_dir}")
    else:
        print(f"Directory exists: {save_dir}")

    print("Inference starts...")
    # Load the best model
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith(".ckpt")]
    if checkpoint_files:
        checkpoint_path = os.path.join(save_dir, checkpoint_files[0])
        print(f"Loading model from {checkpoint_path}")
        model = PEModel.load_from_checkpoint(checkpoint_path, cfg=cfg, task=task, device=device, exp_name=cfg.exp_name)
    else:
        print(f"No checkpoint file found in {save_dir}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        devices=1,
        accelerator="cpu",
        strategy=cfg.trainer.strategy,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
    )

    test_loader = data_module.all_dataloader() # select all samples

    # Extract embeddings
    predictions = trainer.predict(model, test_loader)
    # unzip the list of tuples
    embedding, ids, labels, label_times = zip(*predictions)
    # embeddings is now a tuple of tensors (one per batch)
    # Convert to float32 first if necessary (to avoid BFloat16 issues)
    embedding = [e.float() for e in embedding]

    # Concatenate all batches into a single tensor
    embedding = torch.cat(embedding, dim=0)

    # Convert to NumPy
    embedding = embedding.detach().cpu().numpy()
    ids = torch.cat(ids, dim=0).cpu().numpy().reshape(-1)
    labels = torch.cat(labels, dim=0).cpu().numpy().reshape(-1).astype(bool)
    # flatten list of lists/tuples
    flat_label_times = [np.datetime64(lt) for batch in label_times for lt in batch]
    # turn into numpy column vector of dtype object/string
    label_times = np.array(flat_label_times).reshape(-1)

    # Print shapes
    print(f"Embedding shape: {embedding.shape}")
    print(f"IDs shape: {ids.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label times shape: {label_times.shape}")

    # Save embeddings, ids, labels, and label_times as a pickle file
    data = [embedding, ids, labels, label_times]

    save_path = os.path.join(save_dir, f"deep_featurized_patients.pkl")
    with open(save_path, "wb") as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    main()