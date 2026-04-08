from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_path: str = "dataset/storywell_raw_split_new.csv"
    output_dir: str = "runs/elder_portrait"
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_length: int = 96
    batch_size: int = 8
    epochs: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.2
    event_embed_dim: int = 64
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    max_samples: int = 0
    strict_storywell_schema: bool = True
    event_loss_weight: float = 2.0
    sentiment_loss_weight: float = 1.0
    event_o_weight_scale: float = 1.4
    event_weight_clip_max: float = 8.0
    event_weight_power: float = 0.5
    event_focal_gamma: float = 2.0
    event_sampler_alpha: float = 0.4
    trigger_loss_weight: float = 1.0
    event_type_loss_weight: float = 0.5
    grad_accum_steps: int = 2
    fp16: bool = False
    entity_o_aux_weight: float = 0.4
    decode_objective: str = "precision"
    decode_precision_floor: float = 0.85
    init_checkpoint: str = ""

    def resolve_output_dir(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
