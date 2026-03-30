import hydra
# from hydra.utils import to_absolute_path
import time
import torch
import threading

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

    utils_general.seed_all(cfg.seed)
    task = cfg.task
    device = torch.device(cfg.device.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task: {task}")
    print(f"Experiment name: {cfg.exp_name}")

    # Initialize the DataModule
    data_module = PEDataModule(cfg, task, device)

    # Initialize the model
    model = PEModel(cfg, task, device, cfg.exp_name)

    # Logger
    logger = TensorBoardLogger("tb_logs", name=cfg.exp_name)

    # Callbacks
    #save_dir = to_absolute_path(os.path.join("experiments/report_dir", cfg.exp_name))
    save_dir = os.path.join("/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/outputs", cfg.exp_name) 
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory created: {save_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")
    
    if not os.path.exists(save_dir):
        print(f"Directory was not created: {save_dir}")
    else:
        print(f"Directory exists: {save_dir}")

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
        devices=1,
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
            model = model.load_from_checkpoint(checkpoint_path, cfg=cfg, task=task, device=device, exp_name=cfg.exp_name)
        else:
            print(f"No checkpoint file found in {save_dir}")

        # Testing
        print("Testing starts...")
        test_results = trainer.test(datamodule=data_module, model=model)
        print(test_results)

if __name__ == "__main__":
    main()