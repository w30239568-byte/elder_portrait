import torch
from torch import nn
from transformers import AutoModel


class ElderPortraitMultiTaskModel(nn.Module):
    """
    Joint model:
    - token-level trigger BIO extraction
    - token-level argument role BIO extraction
    - sequence-level event type classification
    - sequence-level sentiment classification
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_event_tags: int,
        num_trigger_tags: int = 0,
        num_event_types: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Avoid background safetensors auto-conversion checks that may call
        # Hugging Face discussions API and raise non-fatal 403 in some repos.
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Linear(hidden_size * 2, num_labels)
        # Keep legacy name `event_classifier` for compatibility. It now predicts
        # argument-role BIO tags (formerly event BIO tags).
        self.event_classifier = nn.Linear(hidden_size, num_event_tags)
        self.trigger_classifier = (
            nn.Linear(hidden_size, num_trigger_tags) if int(num_trigger_tags) > 0 else None
        )
        self.event_type_classifier = (
            nn.Linear(hidden_size * 2, num_event_types)
            if int(num_event_types) > 0
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        cls_hidden = token_hidden[:, 0, :]
        features = torch.cat([cls_hidden, pooled], dim=-1)

        sentiment_logits = self.sentiment_classifier(self.dropout(features))
        event_logits = self.event_classifier(self.dropout(token_hidden))
        outputs = {
            "sentiment_logits": sentiment_logits,
            "event_logits": event_logits,
        }
        if self.trigger_classifier is not None:
            outputs["trigger_logits"] = self.trigger_classifier(self.dropout(token_hidden))
        if self.event_type_classifier is not None:
            outputs["event_type_logits"] = self.event_type_classifier(
                self.dropout(features)
            )
        return outputs


class EventFusionClassifier(nn.Module):
    """
    Legacy v1 model kept for old checkpoint compatibility.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_event_tags: int,
        event_embed_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Avoid background safetensors auto-conversion checks that may call
        # Hugging Face discussions API and raise non-fatal 403 in some repos.
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False)
        hidden_size = self.encoder.config.hidden_size
        self.event_embedding = nn.Embedding(
            num_embeddings=num_event_tags,
            embedding_dim=event_embed_dim,
            padding_idx=0,
        )
        self.event_projection = nn.Linear(event_embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = outputs.last_hidden_state

        event_hidden = self.event_projection(self.event_embedding(event_ids))
        fused_hidden = token_hidden + event_hidden

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (fused_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        cls_hidden = token_hidden[:, 0, :]

        features = torch.cat([cls_hidden, pooled], dim=-1)
        logits = self.classifier(self.dropout(features))
        return logits
