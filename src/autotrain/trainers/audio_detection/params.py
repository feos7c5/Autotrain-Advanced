from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class AudioDetectionParams(AutoTrainParams):
    """
    AudioDetectionParams is a configuration class for audio detection training parameters.
    
    Audio detection identifies specific events in audio and their precise timing (onset/offset).
    Similar to object detection in images, but for temporal events in audio.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the model to be used. Default is "facebook/wav2vec2-base".
        username (Optional[str]): Hugging Face Username.
        lr (float): Learning rate. Default is 3e-5.
        epochs (int): Number of training epochs. Default is 5.
        batch_size (int): Training batch size. Default is 8.
        warmup_ratio (float): Warmup proportion. Default is 0.1.
        gradient_accumulation (int): Gradient accumulation steps. Default is 1.
        optimizer (str): Optimizer to be used. Default is "adamw_torch".
        scheduler (str): Scheduler to be used. Default is "linear".
        weight_decay (float): Weight decay. Default is 0.01.
        max_grad_norm (float): Max gradient norm. Default is 1.0.
        seed (int): Random seed. Default is 42.
        train_split (str): Name of the training data split. Default is "train".
        valid_split (Optional[str]): Name of the validation data split.
        logging_steps (int): Number of steps between logging. Default is -1.
        project_name (str): Name of the project for output directory. Default is "project-name".
        auto_find_batch_size (bool): Whether to automatically find batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision type (fp16, bf16, or None).
        save_total_limit (int): Total number of checkpoints to save. Default is 1.
        token (Optional[str]): Hub Token for authentication.
        push_to_hub (bool): Whether to push the model to the Hugging Face Hub. Default is False.
        eval_strategy (str): Evaluation strategy. Default is "epoch".
        audio_column (str): Name of the audio column in the dataset. Default is "audio".
        events_column (str): Name of the events column in the dataset. Default is "events".
        log (str): Logging method for experiment tracking. Default is "none".
        max_length (int): Maximum audio length in samples. Default is 480000 (30 seconds at 16kHz).
        sampling_rate (int): Target sampling rate for audio. Default is 16000.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
        early_stopping_threshold (float): Minimum change to qualify as an improvement. Default is 0.01.
        event_overlap_threshold (float): IoU threshold for considering two events as overlapping. Default is 0.5.
        confidence_threshold (float): Minimum confidence threshold for event detection. Default is 0.1.
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("facebook/wav2vec2-base", title="Model name")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    lr: float = Field(3e-5, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    batch_size: int = Field(8, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.01, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    train_split: str = Field("train", title="Train split")
    valid_split: Optional[str] = Field(None, title="Validation split")
    logging_steps: int = Field(-1, title="Logging steps")
    project_name: str = Field("project-name", title="Output directory")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    save_total_limit: int = Field(1, title="Save total limit")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    audio_column: str = Field("audio", title="Audio column")
    events_column: str = Field("events", title="Events column")
    log: str = Field("none", title="Logging using experiment tracking")
    max_length: int = Field(480000, title="Maximum audio length in samples (30 seconds at 16kHz)")
    sampling_rate: int = Field(16000, title="Target sampling rate for audio")
    early_stopping_patience: int = Field(5, title="Early stopping patience")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")
    event_overlap_threshold: float = Field(0.5, title="IoU threshold for overlapping events")
    confidence_threshold: float = Field(0.1, title="Minimum confidence threshold for event detection")

    def __post_init__(self):
        """Post-initialization validation."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive") 