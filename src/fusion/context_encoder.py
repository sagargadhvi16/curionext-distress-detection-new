"""Context encoder for metadata embedding."""
import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np


class ContextEncoder(nn.Module):
    """
    Encodes contextual metadata (time of day, location, child age, activity level).
    
    Output: 64-dimensional embedding vector.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        time_of_day_embed_dim: int = 16,
        location_embed_dim: int = 16,
        age_embed_dim: int = 16,
        activity_embed_dim: int = 16
    ):
        """
        Initialize context encoder.
        
        Args:
            embedding_dim: Final embedding dimension (64)
            time_of_day_embed_dim: Embedding dimension for time of day
            location_embed_dim: Embedding dimension for location
            age_embed_dim: Embedding dimension for age
            activity_embed_dim: Embedding dimension for activity level
        """
        super().__init__()
        
        # Time of day embedding (24 hours -> continuous embedding)
        self.time_of_day_encoder = nn.Sequential(
            nn.Linear(1, time_of_day_embed_dim),
            nn.ReLU(),
            nn.Linear(time_of_day_embed_dim, time_of_day_embed_dim)
        )
        
        # Location embedding (categorical -> embedding)
        # Assuming max 10 location types
        self.location_embedding = nn.Embedding(10, location_embed_dim)
        
        # Age embedding (continuous, normalized 0-1)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_embed_dim),
            nn.ReLU(),
            nn.Linear(age_embed_dim, age_embed_dim)
        )
        
        # Activity level embedding (continuous, normalized 0-1)
        self.activity_encoder = nn.Sequential(
            nn.Linear(1, activity_embed_dim),
            nn.ReLU(),
            nn.Linear(activity_embed_dim, activity_embed_dim)
        )
        
        # Combine all embeddings
        total_embed_dim = (
            time_of_day_embed_dim + 
            location_embed_dim + 
            age_embed_dim + 
            activity_embed_dim
        )
        
        # Project to final embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(total_embed_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(
        self,
        time_of_day: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
        child_age: Optional[torch.Tensor] = None,
        activity_level: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode context metadata.
        
        Args:
            time_of_day: Time of day (0-1 normalized, or hour/24)
            location: Location index (0-9, categorical)
            child_age: Child age (0-1 normalized)
            activity_level: Activity level (0-1 normalized)
            
        Returns:
            Context embedding (batch_size, embedding_dim)
        """
        batch_size = None
        
        # Determine batch size from first available input
        for inp in [time_of_day, location, child_age, activity_level]:
            if inp is not None:
                batch_size = inp.shape[0]
                break
        
        if batch_size is None:
            # Default: return zeros
            return torch.zeros(1, 64)
        
        embeddings = []
        
        # Time of day
        if time_of_day is not None:
            time_emb = self.time_of_day_encoder(time_of_day.unsqueeze(-1).float())
        else:
            time_emb = torch.zeros(batch_size, self.time_of_day_encoder[0].out_features)
        embeddings.append(time_emb)
        
        # Location
        if location is not None:
            location_emb = self.location_embedding(location.long())
        else:
            location_emb = torch.zeros(batch_size, self.location_embedding.embedding_dim)
        embeddings.append(location_emb)
        
        # Age
        if child_age is not None:
            age_emb = self.age_encoder(child_age.unsqueeze(-1).float())
        else:
            age_emb = torch.zeros(batch_size, self.age_encoder[0].out_features)
        embeddings.append(age_emb)
        
        # Activity level
        if activity_level is not None:
            activity_emb = self.activity_encoder(activity_level.unsqueeze(-1).float())
        else:
            activity_emb = torch.zeros(batch_size, self.activity_encoder[0].out_features)
        embeddings.append(activity_emb)
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Project to final dimension
        output = self.projection(combined)
        
        return output
    
    def forward_from_dict(self, context_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using dictionary input.
        
        Args:
            context_dict: Dictionary with keys: time_of_day, location, child_age, activity_level
            
        Returns:
            Context embedding
        """
        return self.forward(
            time_of_day=context_dict.get('time_of_day'),
            location=context_dict.get('location'),
            child_age=context_dict.get('child_age'),
            activity_level=context_dict.get('activity_level')
        )

