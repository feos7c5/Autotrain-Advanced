from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class AudioSegmentationParams(AutoTrainParams):
    """
    AudioSegmentationParams is a configuration class for audio segmentation training parameters.
    
    Audio segmentation is similar to token classification but for audio frames instead of text tokens.
    Each audio frame gets assigned a label (e.g., speech, music, silence, speaker_1, etc.).

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the model to use. Default is "facebook/wav2vec2-base".
        lr (float): Learning rate. Default is 3e-5.
        epochs (int): Number of training epochs. Default is 3.
        max_length (int): Maximum audio length in samples. Default is 480000.
        batch_size (int): Training batch size. Default is 8.
        warmup_ratio (float): Warmup proportion. Default is 0.1.
        gradient_accumulation (int): Gradient accumulation steps. Default is 1.
        optimizer (str): Optimizer to use. Default is "adamw_torch".
        scheduler (str): Scheduler to use. Default is "linear".
        weight_decay (float): Weight decay. Default is 0.01.
        max_grad_norm (float): Maximum gradient norm. Default is 1.0.
        seed (int): Random seed. Default is 42.
        train_split (str): Name of the training split. Default is "train".
        valid_split (Optional[str]): Name of the validation split. Default is None.
        audio_column (str): Name of the audio column. Default is "audio".
        tags_column (str): Name of the tags column (frame-level labels). Default is "tags".
        logging_steps (int): Number of steps between logging. Default is -1.
        project_name (str): Name of the project. Default is "project-name".
        auto_find_batch_size (bool): Whether to automatically find the batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision setting (fp16, bf16, or None). Default is None.
        save_total_limit (int): Total number of checkpoints to save. Default is 1.
        token (Optional[str]): Hub token for authentication. Default is None.
        push_to_hub (bool): Whether to push the model to the Hugging Face hub. Default is False.
        eval_strategy (str): Evaluation strategy. Default is "epoch".
        username (Optional[str]): Hugging Face username. Default is None.
        log (str): Logging method for experiment tracking. Default is "none".
        sampling_rate (int): Sampling rate for audio processing. Default is 16000.
        early_stopping_patience (int): Patience for early stopping. Default is 5.
        early_stopping_threshold (float): Threshold for early stopping. Default is 0.01.
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("facebook/wav2vec2-base", title="Model name")
    lr: float = Field(3e-5, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    max_length: int = Field(480000, title="Max audio length in samples")
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
    audio_column: str = Field("audio", title="Audio column")
    tags_column: str = Field("tags", title="Tags column (frame-level labels)")
    logging_steps: int = Field(-1, title="Logging steps")
    project_name: str = Field("project-name", title="Output directory")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    save_total_limit: int = Field(1, title="Save total limit")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    log: str = Field("none", title="Logging using experiment tracking")
    sampling_rate: int = Field(16000, title="Sampling rate for audio processing")
    early_stopping_patience: int = Field(5, title="Early stopping patience")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")

    def __post_init__(self):
        """Post-initialization validation."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive") 