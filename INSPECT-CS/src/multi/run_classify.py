import hydra
import time
import torch

@hydra.main(config_path="./configs", config_name="classify")
def main(cfg):
    import os
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning_model import PEModel
    from datamodule import PEDataModule
    import utils_general

    from torchvision.models import resnet152
    import threading

    # Set environment variable to disable HDF5 file locking
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    utils_general.seed_all(cfg.seed)
    device = torch.device(cfg.device.cuda_device if torch.cuda.is_available() else "cpu")

    # Initialize the DataModule
    data_module = PEDataModule(cfg, device)

    # Initialize the model
    model = PEModel(cfg, device, cfg.exp_name)

    # Logger
    logger = TensorBoardLogger("tb_logs", name=cfg.exp_name)

    # Callbacks
    save_dir = os.path.join(cfg.base_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=1,  # Save only the best model
        monitor=cfg.monitor.metric,
        mode=cfg.monitor.mode,
    )
    callbacks = [lr_monitor, checkpoint_callback]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        devices=cfg.trainer.n_gpus,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        strategy=cfg.trainer.strategy,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        num_sanity_val_steps=0,
    )

    # Training
    if cfg.inference is False:
        print("Training starts...")
        trainer.fit(model, data_module)

        # Testing
        print("Testing starts...")
        test_results = trainer.test(datamodule=data_module, ckpt_path="best")
        print(test_results)
    # Inference
    else:   
        print("Inference starts...")
        # Load the best model
        checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith(".ckpt")]
        if checkpoint_files:
            checkpoint_path = os.path.join(save_dir, checkpoint_files[0])
            print(f"Loading model from {checkpoint_path}")
            model = model.load_from_checkpoint(checkpoint_path, cfg=cfg, device=device, exp_name=cfg.exp_name)
        else:
            print(f"No checkpoint file found in {save_dir}")
        
        # Testing
        print("Testing starts...")
        test_results = trainer.test(datamodule=data_module, model=model)
        print(test_results)

if __name__ == "__main__":
    main()