---
trigger: always_on
description: "study-specific coding patterns, HeteroData usage, and graph neural network best practices"
globs: ["**/*.py", "**/*.ipynb", "src/study/**/*", "experiments/**/*"]
---

# study-Specific Patterns

@context {
  "type": "domain_specific",
  "project": "study",
  "domain": "graph_neural_networks",
  "data_format": "HeteroData",
  "framework": "pytorch_geometric"
}

## HeteroData Usage Patterns

@hetero_data_rules [
  {
    "id": "hetero_data_only",
    "severity": "error",
    "description": "ALL graph operations MUST use PyTorch Geometric's HeteroData format"
  },
  {
    "id": "no_homogeneous_graphs",
    "severity": "error",
    "description": "NEVER use homogeneous tensors for graph operations"
  },
  {
    "id": "convert_to_hetero",
    "severity": "error",
    "description": "ALWAYS convert NetworkX graphs to HeteroData before processing"
  }
]

## Graph Neural Network Patterns

@gnn_patterns [
  {
    "id": "device_agnostic_graphs",
    "description": "Ensure HeteroData works on both CPU and GPU"
  },
  {
    "id": "batch_processing",
    "description": "Use DataLoader for batched graph processing"
  },
  {
    "id": "edge_type_handling",
    "description": "Properly handle multiple edge types in heterogeneous graphs"
  }
]

## Import Patterns

@import_rules [
  {
    "id": "torch_geometric_imports",
    "description": "Use torch_geometric.data.HeteroData for all graph data"
  },
  {
    "id": "study_imports",
    "description": "Use absolute imports from src.study"
  }
]

## Data Processing Patterns

@data_patterns [
  {
    "id": "node_type_consistency",
    "description": "Maintain consistent node type naming across datasets"
  },
  {
    "id": "edge_type_tuples",
    "description": "Use (source_type, relation, target_type) format for edge types"
  },
  {
    "id": "feature_standardization",
    "description": "Standardize node features within each node type"
  }
]

## Model Architecture Patterns

@model_patterns [
  {
    "id": "unified_interfaces",
    "description": "Use unified interfaces for encoders, GNN layers, and tasks"
  },
  {
    "id": "composable_components",
    "description": "Build models from composable, reusable components"
  },
  {
    "id": "multi_task_design",
    "description": "Design all models to support multiple simultaneous tasks"
  }
]

## Environment Variables

@env_vars [
  {
    "var": "study_DATA_DIR",
    "default": "data/",
    "description": "Directory for study datasets"
  },
  {
    "var": "study_MODEL_DIR",
    "default": "models/",
    "description": "Directory for saved models and checkpoints"
  }
]

@version "1.0.0"
@last_updated "2025-01-14"
