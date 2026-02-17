import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class CreditScoringModel(nn.Module):
    """
    Perceptron Multi-capas para Puntaje de Credito
    
    Arquitectura:
    - Capa de Entrada: num_features (depende del preprocesamiento porque se pueden eliminar)
    - Capas Ocultas: Dinamicas
    - Capa de Salida: Linear (-> 1) para producir logits.
    """
    def __init__(self, num_features: int, hidden_layers: List[int],  dropout_rate: float = 0.1, use_batch_norm: bool = True, activation_fn: str = "ReLU"):
        super(CreditScoringModel, self).__init__()
        
        self.num_features = num_features
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_fn_name = activation_fn
        
        layers = []
        input_size = num_features
        
        # network architecture dinamyc
        for i, layer_size in enumerate(hidden_layers):
            
            # lineal layer (neural)
            layers.append(nn.Linear(input_size, layer_size))
            
            # batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))
                
            # activation function
            if activation_fn == "ReLU":
                layers.append(nn.ReLU())
            elif activation_fn == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            elif activation_fn == "GELU":
                layers.append(nn.GELU())
                
            # dropout
            layers.append(nn.Dropout(dropout_rate))
            
            # output_size -> input size
            input_size = layer_size
        
        # output layer
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.network(x)

    def get_model_info(self) -> dict:
        """Get model architecture information"""
        return {
            "model_type": "CreditScoringModel",
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation_fn": self.activation_fn_name,
            "architecture": {
                "input_layer": self.num_features,
                "hidden_layers": self.hidden_layers_config,
                "output_layer": 1
            },
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }