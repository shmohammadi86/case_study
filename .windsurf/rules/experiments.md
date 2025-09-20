---
trigger: always_on
description: "Experimentation guidelines for reproducible research including YAML configs, hyperparameter tuning, and model evaluation"
globs: ["experiments/**/*", "configs/**/*", "scripts/**/*", "**/*.yaml", "**/*.yml", "**/*.py"]
---

# study Experimentation Guidelines

@context {
  "type": "experimentation",
  "project": "study",
  "framework": "pytorch_lightning",
  "tracking": "weights_and_biases",
  "version": "1.0.0"
}

## Canonical Experiment Structure

@structure {
  "required_sections": [
    "config",
    "scripts",
    "logs",
    "results",
    "checkpoints",
    "docs",
    "readme"
  ],
  "configuration_format": "YAML",
  "naming_convention": "descriptive_lowercase_underscore"
}

### Directory Structure

**ALWAYS** scaffold this exact structure for every new experiment:

```
experiments/{EXPERIMENT_NAME}/
├── config/          # Configuration files (YAML only)
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── scripts/         # Execution scripts
│   ├── train.py
│   ├── evaluate.py
│   └── analyze.py
├── logs/           # Training logs and metrics
├── results/        # Results, plots, and analysis
├── checkpoints/    # Model checkpoints
├── docs/           # Experiment documentation
└── README.md       # Experiment description and setup
```

@naming_rules [
  {
    "id": "experiment_names",
    "format": "descriptive_lowercase_underscore",
    "good_examples": ["ogb_mag_benchmark", "protein_interaction_prediction"],
    "bad_examples": ["exp1", "test", "experiment"]
  },
  {
    "id": "script_names",
    "format": "descriptive_purpose",
    "good_examples": ["train_model.py", "evaluate_performance.py", "generate_plots.py"],
    "bad_examples": ["run.py", "script.py", "main.py"]
  }
]

## Configuration Management

@configuration_rules [
  {
    "id": "yaml_only",
    "severity": "error",
    "description": "ALWAYS use YAML format for all configuration"
  },
  {
    "id": "no_hardcoded_params",
    "severity": "error",
    "description": "NEVER hardcode hyperparameters in scripts"
  },
  {
    "id": "config_directory",
    "severity": "error",
    "description": "Store all configs in config/ directory"
  },
  {
    "id": "descriptive_names",
    "severity": "warning",
    "description": "Use descriptive names: model_config.yaml, training_config.yaml"
  }
]

### Configuration Structure Template

**For Distributed GPU Hyperparameter Optimization Experiments:**

```yaml
# config/experiment_config.yaml - Main experiment configuration
name: "Experiment Name"
description: "Detailed experiment description"
version: "1.0.0"
tags: ["tag1", "tag2"]

# Hierarchical hyperparameter spaces
hyperparameters:
  scientific:  # Architectural choices (grid search)
    model.hidden_dim:
      type: choice
      choices: [128, 256, 512]
    model.num_layers:
      type: choice
      choices: [2, 3, 4]
      
  nuisance:  # Training details (Bayesian optimization)
    training.learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-2
    training.batch_size:
      type: choice
      choices: [16, 32, 64]

# Optimization strategy
optimization:
  max_scientific_trials: 9   # Grid combinations
  max_nuisance_trials: 25    # Bayesian trials per scientific config
  scientific_strategy: grid
  nuisance_strategy: optuna
  num_gpus: 8
  timeout_hours: 24

# Data configuration
data:
  network_path: "data/network.pkl"
  embeddings_path: "data/embeddings.parquet"
  batch_size: 32
  num_workers: 4

# Experiment tracking
tracking:
  project: "project-name"
  group: "experiment-group"
  tags: ["distributed", "hyperopt"]
```

## Hyperparameter Classification

@hyperparameter_types {
  "scientific": {
    "description": "Parameters that directly affect model performance and should be tuned",
    "examples": [
      "learning_rate",
      "batch_size",
      "model_architecture",
      "dropout_rate",
      "weight_decay",
      "optimizer_settings",
      "scheduler_parameters"
    ]
  },
  "nuisance": {
    "description": "Parameters that affect training but not core model performance",
    "examples": [
      "max_epochs",
      "early_stopping_patience",
      "logging_frequency",
      "checkpoint_intervals",
      "random_seeds"
    ]
  },
  "fixed": {
    "description": "Parameters that should remain constant across experiments",
    "examples": [
      "dataset_splits",
      "evaluation_metrics",
      "hardware_settings",
      "device_precision",
      "infrastructure_paths"
    ]
  }
}

## Tuning Methodology

@tuning_strategy {
  "approach": "systematic",
  "steps": [
    "start_simple",
    "tune_one_at_a_time",
    "use_search_strategy",
    "track_and_reproduce",
    "report_standards"
  ]
}

### 1. Start Simple

@rules [
  {
    "id": "baseline_first",
    "description": "Begin with baseline model and default hyperparameters"
  },
  {
    "id": "performance_floor",
    "description": "Establish performance floor before optimization"
  },
  {
    "id": "document_baseline",
    "description": "Document baseline results and configuration"
  }
]

### 2. Hierarchical Optimization Strategy

@hierarchical_rules [
  {
    "id": "scientific_first",
    "severity": "error",
    "description": "ALWAYS optimize scientific hyperparameters (architecture) first via grid search"
  },
  {
    "id": "nuisance_second",
    "severity": "error", 
    "description": "Optimize nuisance hyperparameters (training) second via Bayesian optimization"
  },
  {
    "id": "parameter_classification",
    "description": "Classify parameters as scientific (affects model capacity) vs nuisance (affects training)"
  },
  {
    "id": "document_hierarchy",
    "description": "Document the rationale for hyperparameter classification"
  }
]

**Scientific Parameters (Grid Search):**
- Model architecture (layers, dimensions)
- Activation functions
- Regularization structure

**Nuisance Parameters (Bayesian Optimization):**
- Learning rates and schedules
- Batch sizes
- Optimizer settings
- Early stopping criteria

### 3. Search Strategy

@search_phases [
  {
    "phase": "coarse_search",
    "description": "Wide range, few points",
    "purpose": "Find general region of good parameters"
  },
  {
    "phase": "fine_search",
    "description": "Narrow range around best coarse results",
    "purpose": "Refine parameter values"
  },
  {
    "phase": "final_validation",
    "description": "Best configuration on held-out test set",
    "purpose": "Validate generalization"
  }
]

### 4. Tracking and Reproducibility

@tracking_requirements [
  {
    "id": "log_all_params",
    "severity": "error",
    "description": "ALWAYS log all hyperparameters and results"
  },
  {
    "id": "experiment_tracking",
    "severity": "error",
    "description": "Use experiment tracking tools (MLflow, Weights & Biases)"
  },
  {
    "id": "save_exact_config",
    "severity": "error",
    "description": "Save exact configuration for reproducible results"
  },
  {
    "id": "include_seeds",
    "severity": "error",
    "description": "Include random seeds in all configurations"
  }
]

### 5. Reporting Standards

@reporting_requirements [
  {
    "id": "document_methodology",
    "description": "Document search space and methodology"
  },
  {
    "id": "statistical_significance",
    "description": "Report statistical significance of improvements"
  },
  {
    "id": "confidence_intervals",
    "description": "Include confidence intervals where applicable"
  },
  {
    "id": "comparison_tables",
    "description": "Provide comparison tables and visualizations"
  }
]

## Distributed GPU Hyperparameter Optimization

@distributed_gpu_requirements [
  {
    "id": "use_distributed_executor",
    "severity": "error", 
    "description": "ALWAYS use DistributedGPUExecutor for multi-GPU hyperparameter optimization"
  },
  {
    "id": "hierarchical_optimization",
    "severity": "error",
    "description": "Use 2-layer hierarchical optimization: scientific (grid) + nuisance (Bayesian)"
  },
  {
    "id": "shared_data_loading",
    "severity": "warning",
    "description": "Use SharedDataManager to minimize I/O overhead across trials"
  },
  {
    "id": "gpu_load_balancing",
    "severity": "warning",
    "description": "Implement dynamic GPU load balancing with work-stealing queues"
  }
]

### Usage Pattern for Distributed Experiments

```python
# scripts/run_hyperopt.py
from study.executors.experiments import NBFNetHyperoptExecutor

executor = NBFNetHyperoptExecutor(
    config_path="config/experiment_config.yaml",
    num_gpus=8,
    output_dir="results/"
)

results = executor.run()
```

### Configuration Requirements

@config_schema {
  "required_sections": [
    "hyperparameters.scientific",
    "hyperparameters.nuisance", 
    "optimization",
    "data",
    "tracking"
  ],
  "hyperparameter_types": {
    "choice": "Discrete options",
    "uniform": "Continuous uniform distribution",
    "loguniform": "Log-scale uniform distribution",
    "int": "Integer range"
  },
  "optimization_strategies": {
    "scientific": ["grid", "random"],
    "nuisance": ["optuna", "random", "ray"]
  }
}

## study-Specific Requirements

@study_rules [
  {
    "id": "executors_layer_integration",
    "severity": "error",
    "description": "ALWAYS use src/study/executors for experiment orchestration"
  },
  {
    "id": "no_bypass_executors",
    "severity": "error",
    "description": "NEVER bypass the executors layer for hyperparameter optimization"
  },
  {
    "id": "use_executor_manager",
    "severity": "error",
    "description": "Use ExecutorManager for coordinating multiple experiments"
  }
]

### 4-Layer Architecture Compliance

@architecture_requirements [
  {
    "id": "data_layer",
    "requirement": "Use studyDataModule or UnifiedDataModule"
  },
  {
    "id": "modules_layer",
    "requirement": "Use composable encoders, GNN layers, and tasks"
  },
  {
    "id": "models_layer",
    "requirement": "Use UnifiedstudyModel or specialized models"
  },
  {
    "id": "executors_layer",
    "requirement": "Use ExecutorManager for orchestration"
  }
]

### HeteroData Requirements

@data_format_rules [
  {
    "id": "hetero_data_format",
    "severity": "error",
    "description": "ALWAYS use HeteroData format for all graph operations"
  },
  {
    "id": "format_conversion",
    "description": "Convert other formats (NetworkX, homogeneous) to HeteroData"
  },
  {
    "id": "data_validation",
    "description": "Validate data format before model training"
  }
]

## Experiment Documentation

### README.md Template

```markdown
# Experiment: {EXPERIMENT_NAME}

## Objective
Brief description of what this experiment aims to achieve.

## Methodology
- Model architecture and key components
- Hyperparameter search strategy
- Evaluation metrics and validation approach

## Configuration
- Link to configuration files
- Key hyperparameters and their ranges
- Fixed parameters and assumptions

## Results
- Summary of key findings
- Performance metrics and comparisons
- Statistical significance tests

## Reproducibility
- Exact commands to reproduce results
- Environment requirements
- Random seeds and version information

## Analysis
- Interpretation of results
- Limitations and future work
- Lessons learned
```

@documentation_requirements [
  {
    "id": "comprehensive_readme",
    "severity": "error",
    "description": "ALWAYS create comprehensive README for each experiment"
  },
  {
    "id": "document_assumptions",
    "description": "Document all assumptions and limitations"
  },
  {
    "id": "include_examples",
    "description": "Include code examples and usage instructions"
  },
  {
    "id": "troubleshooting_guides",
    "description": "Provide troubleshooting guides for common issues"
  }
]

## Reproducibility Standards

@reproducibility_requirements {
  "environment": {
    "versions": "Pin exact dependency versions",
    "lock_files": "Use uv.lock for reproducible environments",
    "documentation": "Document Python version and system requirements",
    "hardware": "Include hardware specifications for GPU experiments"
  },
  "random_seeds": {
    "all_generators": "Set random seeds for all random number generators",
    "split_seeds": "Use different seeds for train/validation/test splits",
    "config_documentation": "Document seed values in configuration files",
    "multiple_runs": "Use seed arrays for multiple runs"
  },
  "data_versioning": {
    "version_datasets": "Version datasets and preprocessing steps",
    "checksums": "Use data checksums to verify integrity",
    "source_documentation": "Document data sources and collection methods",
    "transformation_tracking": "Track data transformations and filtering steps"
  },
  "code_versioning": {
    "tag_versions": "Tag code versions for published results",
    "commit_hashes": "Use git commit hashes in experiment logs",
    "archive_snapshots": "Archive code snapshots for important experiments",
    "document_modifications": "Document any manual modifications or patches"
  }
}

## Experiment Lifecycle

@lifecycle_phases [
  {
    "phase": "planning",
    "steps": [
      "Define clear objectives and success criteria",
      "Design experiment methodology and controls",
      "Estimate computational requirements",
      "Plan evaluation and analysis approach"
    ]
  },
  {
    "phase": "execution",
    "steps": [
      "Set up experiment directory structure",
      "Configure logging and monitoring",
      "Run baseline experiments first",
      "Execute hyperparameter search systematically"
    ]
  },
  {
    "phase": "analysis",
    "steps": [
      "Collect and organize all results",
      "Perform statistical analysis",
      "Generate visualizations and reports",
      "Document findings and conclusions"
    ]
  },
  {
    "phase": "documentation",
    "steps": [
      "Write comprehensive experiment report",
      "Archive all code, data, and results",
      "Share findings with team",
      "Plan follow-up experiments"
    ]
  }
]

## Integration Requirements

@integration_rules [
  {
    "id": "component_testing",
    "description": "ALWAYS test new components in isolation first"
  },
  {
    "id": "interface_patterns",
    "description": "Ensure components follow unified interface patterns"
  },
  {
    "id": "architecture_compatibility",
    "description": "Validate compatibility with existing architecture"
  },
  {
    "id": "api_documentation",
    "description": "Document component APIs and usage examples"
  }
]

### Multi-Task Experiments

@multi_task_requirements [
  {
    "id": "multiple_tasks_support",
    "severity": "error",
    "description": "ALWAYS design experiments to support multiple tasks"
  },
  {
    "id": "task_manager_usage",
    "description": "Use TaskManager for coordinating multiple objectives"
  },
  {
    "id": "report_all_tasks",
    "description": "Report results for all tasks and their combinations"
  },
  {
    "id": "analyze_interactions",
    "description": "Analyze task interactions and trade-offs"
  }
]

### Benchmarking Standards

@benchmarking_standards [
  {
    "id": "baseline_comparison",
    "severity": "error",
    "description": "ALWAYS compare against established baselines"
  },
  {
    "id": "standard_datasets",
    "description": "Use standard datasets and evaluation protocols"
  },
  {
    "id": "computational_efficiency",
    "description": "Report computational efficiency metrics"
  },
  {
    "id": "scaling_analysis",
    "description": "Include scaling analysis for large graphs"
  }
]

## Execution and Monitoring

### Command Line Execution

```bash
# Standard execution
python scripts/run_hyperopt.py

# With custom parameters
python -m study.executors.experiments.nbfnet_hyperopt_executor \
    config/experiment_config.yaml \
    --num-gpus 8 \
    --output-dir results/ \
    --debug
```

### Monitoring and Logging

@monitoring_requirements [
  {
    "id": "wandb_integration",
    "severity": "error",
    "description": "ALWAYS use wandb for experiment tracking with meaningful run names"
  },
  {
    "id": "gpu_monitoring",
    "severity": "warning",
    "description": "Monitor GPU utilization and memory usage during optimization"
  },
  {
    "id": "progress_tracking",
    "severity": "warning",
    "description": "Track optimization progress with trial completion rates"
  },
  {
    "id": "error_logging",
    "severity": "error",
    "description": "Log all errors and failed trials with detailed diagnostics"
  }
]

### Result Analysis

```python
# Analyze optimization results
import yaml

with open('results/optimization_results.yaml', 'r') as f:
    results = yaml.safe_load(f)
    
print(f"Best score: {results['best_value']:.6f}")
print(f"Best config: {results['best_overall_config']}")
print(f"Total time: {results['total_time']:.2f}s")
print(f"GPU efficiency: {results['gpu_efficiency']:.2%}")
```

## Gene Embeddings Hyperparameter Optimization (2025-01-28)

### Archive Management

@archive_strategy {
  "date": "2025-01-28",
  "action": "archived_old_experiments",
  "source": ["nbfnet_contrastive_hyperoptim", "enhanced_nbfnet_contrastive_hyperoptim"],
  "destination": "archive/2025-01-28_old_hyperoptim/",
  "reason": "interface_completely_changed"
}

### Tutorial Integration Experiment

@experiment_definition {
  "name": "gene_embeddings_hyperoptim",
  "objective": "Learn gene embeddings with k*NN score > 4.0 (beating baseline 4.4)",
  "integration": "ALL_tutorial_components_1_through_6",
  "approach": "hierarchical_hyperparameter_optimization"
}

#### Tutorial Component Mapping

@tutorial_integration {
  "tutorial_1": {
    "component": "studyDataset/DataModule",
    "implementation": "create_synthetic_gene_dataset()",
    "purpose": "biological_data_loading",
    "network_statistics": "compute_network_statistics() - comprehensive network analysis"
  },
  "tutorial_2": {
    "component": "Unified_Encoder_Interface", 
    "implementation": "GeneEncoder (categorical/ontological/pretrained)",
    "purpose": "gene_encoding_strategies"
  },
  "tutorial_3": {
    "component": "study_Backbone",
    "implementation": "studyBackbone (NBFNet-style)",
    "purpose": "network_learning_architecture"
  },
  "tutorial_4": {
    "component": "TaskManager",
    "implementation": "GeneTaskManager (5 biological tasks)",
    "purpose": "multitask_biological_learning"
  },
  "tutorial_5": {
    "component": "Complete_LEGO_Model",
    "implementation": "GeneEmbeddingLEGOModel",
    "purpose": "end_to_end_architecture"
  },
  "tutorial_6": {
    "component": "Scaling_via_Executor",
    "implementation": "SimpleBayesianOptimizer + distributed_jobs",
    "purpose": "hierarchical_hyperparameter_optimization"
  }
}

#### Gene-Centric Biological Tasks

@biological_tasks [
  {
    "task_name": "go_function",
    "type": "multi_label_classification",
    "output_dim": 100,
    "loss": "bce",
    "description": "Gene Ontology function prediction"
  },
  {
    "task_name": "pathway_membership", 
    "type": "multi_label_classification",
    "output_dim": 50,
    "loss": "bce",
    "description": "Biological pathway membership"
  },
  {
    "task_name": "disease_association",
    "type": "multi_label_classification", 
    "output_dim": 30,
    "loss": "focal",
    "description": "Disease gene associations"
  },
  {
    "task_name": "tissue_expression",
    "type": "regression",
    "output_dim": 20,
    "loss": "huber", 
    "description": "Tissue-specific gene expression"
  },
  {
    "task_name": "gene_interaction",
    "type": "binary_classification",
    "output_dim": 1,
    "loss": "bce",
    "description": "Gene-gene interaction prediction"
  }
]

#### Hierarchical Hyperparameter Strategy

@hyperparameter_hierarchy {
  "scientific_hyperparams": {
    "optimization": "grid_search",
    "dimensions": 6,
    "parameters": [
      "gene_encoder_type",
      "gene_embedding_dim", 
      "backbone_depth",
      "backbone_width",
      "projection_dim",
      "contrastive_weight"
    ],
    "search_space": "3×3×4×3×3×3 = 972 combinations (limited to 8-16 for efficiency)"
  },
  "nuisance_hyperparams": {
    "optimization": "bayesian_optimization",
    "dimensions": 4,
    "parameters": [
      "learning_rate",
      "dropout_rate", 
      "weight_decay",
      "temperature"
    ],
    "optimizer": "SimpleBayesianOptimizer with 10 random starts"
  }
}

#### k*NN Biological Evaluation

@kstar_metric {
  "purpose": "biological_quality_assessment",
  "method": "cross_validation_on_functional_annotations",
  "target_score": 4.0,
  "baseline_score": 4.4,
  "cv_folds": 5,
  "distance_metric": "euclidean",
  "correlation_type": "distance_matrix_correlation",
  "scaling_factor": "L_C = 1.0",
  "adaptive_k": "sqrt(n_train) capped at 50"
}

#### Production Deployment Strategy

@production_scaling {
  "real_data": {
    "gene_network": "STRING_human_19000_genes",
    "annotations": "real_GO_annotations_10000_terms",
    "pathways": "KEGG_Reactome_500_pathways",
    "diseases": "DisGeNET_associations",
    "expression": "GTEx_tissue_expression"
  },
  "computational_resources": {
    "gpus": "4-8_multi_GPU_training",
    "jobs": "16_concurrent_hyperparameter_jobs",
    "cluster": "SLURM_Kubernetes_deployment",
    "tracking": "MLflow_experiment_management"
  },
  "target_performance": {
    "kstar_score": "> 4.0 (beating baseline 4.4)",
    "go_function_auc": "> 0.85",
    "pathway_membership_ap": "> 0.75",
    "gene_interaction_auc": "> 0.80"
  }
}

#### File Organization

@file_structure {
  "main_script": "run_gene_embeddings_hyperoptim.py (754 lines)",
  "configuration": "config.yaml (comprehensive hyperparameter spaces)",
  "documentation": "README.md (detailed usage and biological context)",
  "results": "outputs/gene_embedding_results.json",
  "validation": "runs_successfully_with_synthetic_data"
}

@experiment_success_criteria [
  "complete_tutorial_integration",
  "hierarchical_hyperparameter_optimization", 
  "biological_evaluation_framework",
  "production_ready_architecture",
  "target_kstar_score_achievement"
]

## studyDataset Loading Guidelines

@study_loading_rules [
  {
    "id": "standard_loading_pattern",
    "description": "ALWAYS use the standard 5-step loading pattern",
    "pattern": "1. load_gene_network() 2. studyDataset.from_networkx() 3. load_bio_registry() 4. add_gene_embeddings() 5. add_gene_benchmarks()"
  },
  {
    "id": "network_statistics", 
    "description": "ALWAYS compute network statistics for analysis",
    "method": "dataset.compute_network_statistics(compute_shortest_paths=True)"
  },
  {
    "id": "bioregistry_mapping",
    "description": "ALWAYS use bioregistry for gene ID translation",
    "primary_index": "ensembl_gene_id"
  },
  {
    "id": "execution_command",
    "description": "ALWAYS use uv run --frozen python for script execution"
  }
]

@available_networks [
  "functional", "phenotypic", "molecular", "predicted",
  "FunMol", "FunMolPred", "FunPhenMol", "FunPhenMolPred", 
  "STRING"
]

@version "1.3.0"
@last_updated "2025-01-28"
