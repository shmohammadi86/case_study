---
trigger: always_on
description: "Machine learning best practices for PyTorch, PyTorch Lightning, experiment management, and model development"
globs: ["**/*.py", "**/*.ipynb", "experiments/**/*", "models/**/*", "src/**/*"]
---

# Machine Learning Best Practices

@context {
  "type": "machine_learning",
  "primary_frameworks": ["pytorch", "pytorch_lightning", "scikit_learn"],
  "domain": "deep_learning_and_classical_ml",
  "model_types": ["neural_networks", "traditional_ml"]
}

## PyTorch Best Practices

@pytorch_rules [
  {
    "id": "device_agnostic_code",
    "severity": "error",
    "description": "Write device-agnostic code that works on both CPU and GPU"
  },
  {
    "id": "proper_tensor_dtypes",
    "description": "Use appropriate tensor dtypes for memory efficiency"
  },
  {
    "id": "avoid_item_calls",
    "description": "Avoid .item() calls in training loops for performance"
  },
  {
    "id": "gradient_management",
    "severity": "error",
    "description": "Properly manage gradients and autograd context"
  },
  {
    "id": "memory_efficiency",
    "description": "Use gradient accumulation and mixed precision for large models"
  }
]

### PyTorch Memory Management

- Use `torch.no_grad()` for inference to save memory
- Clear gradients with `optimizer.zero_grad()` before backward pass
- Use `torch.cuda.empty_cache()` to free GPU memory when needed
- Leverage gradient checkpointing for memory-intensive models

## PyTorch Lightning Best Practices

@lightning_rules [
  {
    "id": "lightning_module_inheritance",
    "severity": "error",
    "description": "ALL models MUST inherit from pl.LightningModule"
  },
  {
    "id": "structured_training_loops",
    "description": "Use Lightning's structured training/validation/test loops"
  },
  {
    "id": "proper_logging",
    "description": "Use Lightning's logging capabilities for metrics and artifacts"
  },
  {
    "id": "checkpoint_management",
    "description": "Implement proper checkpointing and model saving"
  },
  {
    "id": "distributed_training",
    "description": "Design models to work with distributed training"
  }
]

### Lightning Module Structure

- Implement `training_step()`, `validation_step()`, `test_step()`
- Use `configure_optimizers()` for optimizer setup
- Leverage `LightningDataModule` for data handling
- Use callbacks for training customization

## Scikit-learn Best Practices

@sklearn_rules [
  {
    "id": "pipeline_usage",
    "severity": "error",
    "description": "ALWAYS use pipelines for preprocessing and modeling"
  },
  {
    "id": "cross_validation",
    "severity": "error",
    "description": "Use proper cross-validation for model evaluation"
  },
  {
    "id": "data_leakage_prevention",
    "severity": "error",
    "description": "Prevent data leakage by fitting preprocessors only on training data"
  },
  {
    "id": "hyperparameter_tuning",
    "description": "Use grid search or random search for hyperparameter optimization"
  }
]

## Model Development Workflow

@workflow_rules [
  {
    "id": "baseline_first",
    "severity": "error",
    "description": "ALWAYS establish a simple baseline before complex models"
  },
  {
    "id": "iterative_development",
    "description": "Develop models iteratively with small improvements"
  },
  {
    "id": "data_understanding",
    "severity": "error",
    "description": "Thoroughly understand data before modeling"
  },
  {
    "id": "evaluation_strategy",
    "severity": "error",
    "description": "Define evaluation strategy before training"
  }
]

## Data Handling

@data_rules [
  {
    "id": "train_val_test_split",
    "severity": "error",
    "description": "ALWAYS use proper train/validation/test splits"
  },
  {
    "id": "data_augmentation",
    "description": "Use data augmentation to improve generalization"
  },
  {
    "id": "feature_engineering",
    "description": "Apply domain knowledge in feature engineering"
  },
  {
    "id": "data_versioning",
    "description": "Version datasets and track data lineage"
  }
]

## Model Evaluation

@evaluation_rules [
  {
    "id": "multiple_metrics",
    "description": "Use multiple evaluation metrics appropriate for the task"
  },
  {
    "id": "statistical_significance",
    "description": "Test statistical significance of model improvements"
  },
  {
    "id": "cross_validation",
    "severity": "error",
    "description": "Use cross-validation for robust performance estimates"
  },
  {
    "id": "error_analysis",
    "description": "Perform thorough error analysis on failed predictions"
  }
]

## Experiment Tracking

@tracking_rules [
  {
    "id": "reproducible_experiments",
    "severity": "error",
    "description": "ALWAYS make experiments reproducible with proper seeding"
  },
  {
    "id": "hyperparameter_logging",
    "severity": "error",
    "description": "Log all hyperparameters and configuration"
  },
  {
    "id": "artifact_management",
    "description": "Save models, metrics, and important artifacts"
  },
  {
    "id": "experiment_documentation",
    "description": "Document experiment objectives and results"
  }
]

## Model Deployment

@deployment_rules [
  {
    "id": "model_serialization",
    "description": "Use proper model serialization for deployment"
  },
  {
    "id": "inference_optimization",
    "description": "Optimize models for inference performance"
  },
  {
    "id": "model_monitoring",
    "description": "Implement model performance monitoring in production"
  },
  {
    "id": "version_management",
    "description": "Implement proper model versioning and rollback strategies"
  }
]

## Deep Learning Specific

@deep_learning_rules [
  {
    "id": "appropriate_initialization",
    "description": "Use appropriate weight initialization schemes"
  },
  {
    "id": "learning_rate_scheduling",
    "description": "Implement learning rate scheduling for training stability"
  },
  {
    "id": "regularization_techniques",
    "description": "Apply appropriate regularization techniques"
  },
  {
    "id": "batch_normalization",
    "description": "Use batch normalization or layer normalization appropriately"
  }
]

## Performance Optimization

@optimization_rules [
  {
    "id": "profile_training",
    "description": "Profile training loops to identify bottlenecks"
  },
  {
    "id": "efficient_data_loading",
    "description": "Optimize data loading with appropriate num_workers"
  },
  {
    "id": "mixed_precision",
    "description": "Use mixed precision training for faster training"
  },
  {
    "id": "model_compilation",
    "description": "Use model compilation when available (torch.compile)"
  }
]

## Ethics and Bias

@ethics_rules [
  {
    "id": "bias_assessment",
    "description": "Assess models for potential bias and fairness issues"
  },
  {
    "id": "data_privacy",
    "description": "Ensure data privacy and compliance with regulations"
  },
  {
    "id": "interpretability",
    "description": "Provide model interpretability where required"
  },
  {
    "id": "responsible_ai",
    "description": "Follow responsible AI practices and guidelines"
  }
]

@version "1.0.0"
@last_updated "2025-01-14"
