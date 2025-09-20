---
trigger: always_on
description: "Project management guidelines including cookiecutter workflows, dependency management, and configuration standards"
globs: ["pyproject.toml", "pixi.toml", ".cruft.*", "cookiecutter.*", "noxfile.py", "**/*.toml", "**/*.yaml", "**/*.yml"]
---

# Project Management Guidelines

@context {
  "type": "project_management",
  "language": "python",
  "package_manager": "uv",
  "task_runner": "nox",
  "project_type": "scientific_library",
  "template_system": "cookiecutter_cruft"
}

## 🔥 CRITICAL: Cookiecutter Template File Management

@cookiecutter_workflow [
  {
    "id": "template_first_rule",
    "severity": "error",
    "description": "CRITICAL: For ANY changes to cookiecutter template files, ALWAYS make changes in cookiecutter template first, then use cruft update"
  },
  {
    "id": "no_direct_template_edits",
    "severity": "error",
    "description": "NEVER edit cookiecutter template files directly in projects - this breaks consistency across all projects"
  },
  {
    "id": "cruft_workflow",
    "severity": "error",
    "description": "Workflow: 1) Edit in cookiecutter template, 2) Git commit/push template, 3) cruft update in project"
  },
  {
    "id": "template_location",
    "severity": "error",
    "description": "Cookiecutter template is ALWAYS located at scratch/repos/cookiecutter/ - NEVER use find command to locate it"
  }
]

### Cookiecutter Template Files (NEVER edit directly in projects)

@template_files {
  "core_config": [
    ".black.toml", ".coveragerc", ".cruft.json", ".cruft.toml", ".cz.toml",
    ".gitignore", ".gitlab-ci.yml", ".isort.cfg", ".marimo/config.toml",
    ".mypy.ini", ".pre-commit-config.yaml", ".pytest.ini", ".python-version",
    ".readthedocs.yaml", ".ruff.toml", ".yamllint", "pyproject.toml",
    "noxfile.py", "pixi.toml", "README.md"
  ],
  "containers": [
    "Dockerfile", ".devcontainer/Dockerfile", ".devcontainer/devcontainer.json", ".dockerignore"
  ],
  "documentation": [
    "docs/ (entire directory and all subdirectories)"
  ],
  "dependencies": [
    "locks/ (entire directory)", "requirements/ (entire directory)"
  ],
  "source_template": [
    "src/study/ (template structure)", "tests/test_study.py"
  ],
  "ide_structure": [
    ".vscode/ (directory)", "experiments/README.md", "notebooks/README.md", "scratch/cache/ structure",
    ".cursor/rules/ (entire directory)"
  ]
}

## 🛡️ Critical Safety Rules

@bash_safety_rules [
  {
    "id": "no_exclamation_marks",
    "severity": "error",
    "description": "NEVER use exclamation marks (!) in bash commands - they trigger bash history expansion and break commands"
  },
  {
    "id": "destructive_command_permission",
    "severity": "error",
    "description": "CLI commands such as 'rm -rf' should NEVER be run without explicit user permission"
  },
  {
    "id": "quote_strings",
    "description": "Use single quotes around strings that contain exclamation marks or avoid them entirely"
  }
]

@user_preferences [
  {
    "id": "autonomous_debugging",
    "description": "User prefers assistant to be autonomous and iteratively debug and test solutions"
  },
  {
    "id": "no_find_command",
    "severity": "error",
    "description": "NEVER use find command - user has explicitly forbidden it multiple times"
  }
]

## 🔒 Special Folders Requiring Special Treatment

@special_folders {
  "version_control": [".git/", ".gitmodules"],
  "virtual_environments": [".venv/", "venv/", "env/", ".env/", ".pixi/", "node_modules/"],
  "python_cache": ["__pycache__/", "*.pyc", "*.py[cod]", "build/", "dist/", ".eggs/", "*.egg-info/", "*.egg"],
  "tool_caches": [".nox/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".cache/", ".ipynb_checkpoints/"],
  "test_coverage": [".coverage", ".coverage.*", "test-reports/", "htmlcov/"],
  "ml_artifacts": ["wandb/", "lightning_logs/", "dask-worker-space/", "catboost_info/", "experiments/**/checkpoints/**", "experiments/**/logs/**", "experiments/**/data/**"],
  "ide_configs": [".idea/", ".vscode/", ".cursor/", ".devcontainer/"],
  "working_directories": ["scratch/", "temp/", "__marimo__/", "layouts/", "notebooks/figures/", "notebooks/output/", "experiments/scratch/", "experiments/old/"],
  "generated_files": ["*.log", "*.ckpt", "*.pkl", "*.h5ad", "*.h5mu", "docs/slides/"],
  "system_files": [".DS_Store", ".nfs*", "local_*"],
  "lock_files": ["uv.lock", "poetry.lock", "Pipfile.lock"]
}

## ⚡ Command Operation Guidelines

@command_exclusions [
  {
    "id": "standard_exclusions",
    "severity": "error",
    "description": "ALWAYS exclude special folders in find, grep, diff, cp, rsync commands"
  },
  {
    "id": "exclusion_pattern",
    "description": "Use: --exclude='.git' --exclude='.nox' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='.pixi' --exclude='scratch' --exclude='temp' --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='wandb' --exclude='lightning_logs' --exclude='.ipynb_checkpoints'"
  },
  {
    "id": "forbidden_commands",
    "severity": "error",
    "description": "NEVER use find command - it's explicitly forbidden by user"
  }
]

@file_operation_rules [
  {
    "id": "template_file_detection",
    "severity": "error",
    "description": "Before modifying any file, check if it's a cookiecutter template file using the template_files list"
  },
  {
    "id": "working_directory_safety",
    "severity": "error",
    "description": "NEVER run destructive commands (rm -rf) in working directories without explicit permission"
  },
  {
    "id": "cache_directory_handling",
    "description": "Cache directories can be safely deleted for cleanup but should be excluded from operations"
  }
]

## 🏗️ Architecture Compliance (for study projects)

@architecture_rules [
  {
    "id": "hetero_data_format",
    "severity": "error",
    "description": "ALL graph operations MUST use PyTorch Geometric's HeteroData format"
  },
  {
    "id": "four_layer_architecture",
    "severity": "error",
    "description": "ALWAYS follow 4-layer architecture: data/modules/models/tuning layers"
  },
  {
    "id": "no_homogeneous_graphs",
    "severity": "error",
    "description": "NEVER use homogeneous tensors for graph operations"
  },
  {
    "id": "multi_task_design",
    "severity": "error",
    "description": "ALL models must support multiple simultaneous tasks"
  }
]

## Package Management

@package_management {
  "primary_tool": "uv",
  "config_file": "pyproject.toml",
  "lock_file": "uv.lock",
  "virtual_env": "uv venv"
}

### UV Package Manager Rules

@uv_rules [
  {
    "id": "use_uv_exclusively",
    "severity": "error",
    "description": "ALWAYS use uv for dependency management when package_manager is uv"
  },
  {
    "id": "no_pip_direct",
    "severity": "error",
    "description": "NEVER use pip directly - use uv add, uv remove, uv sync"
  },
  {
    "id": "lock_file_usage",
    "severity": "error",
    "description": "Use uv.lock for reproducible builds"
  },
  {
    "id": "uv_with_nox",
    "description": "Use uv through nox for environment isolation when possible"
  }
]

## Development Workflow

@development_rules [
  {
    "id": "nox_sessions",
    "severity": "error",
    "description": "ALWAYS use nox sessions for formatting, linting, testing"
  },
  {
    "id": "ruff_mypy_tools",
    "severity": "error",
    "description": "Use ruff for formatting/linting, mypy for type checking"
  },
  {
    "id": "experiment_structure",
    "description": "Use canonical experiment structure with config/, scripts/, logs/, results/, checkpoints/"
  }
]

## Project Structure

@structure_rules [
  {
    "id": "src_layout",
    "severity": "error",
    "description": "ALWAYS use src/ layout for Python packages"
  },
  {
    "id": "standard_directories",
    "severity": "warning",
    "description": "Use standard directories: tests/, docs/, experiments/, notebooks/"
  },
  {
    "id": "pyproject_toml",
    "severity": "error",
    "description": "ALWAYS configure project in pyproject.toml"
  },
  {
    "id": "scratch_organization",
    "description": "Use scratch/ for tool caches, temp files, and workspace isolation"
  }
]

### Standard Project Layout

```
study/
├── src/
│   └── study/
├── tests/
├── docs/
├── experiments/
├── notebooks/
├── scratch/           # Tool caches and temp workspace
│   ├── cache/        # Tool-specific caches
│   ├── temp/         # Temporary files
│   └── repos/        # Local repositories (like cookiecutter template)
├── pyproject.toml    # ⚠️ COOKIECUTTER TEMPLATE FILE
└── README.md         # ⚠️ COOKIECUTTER TEMPLATE FILE
```

## Environment Configuration

@environment_rules [
  {
    "id": "env_var_prefix",
    "description": "Use study_* prefix for environment variables"
  },
  {
    "id": "config_defaults",
    "description": "Provide sensible defaults for all configuration"
  },
  {
    "id": "config_validation",
    "severity": "error",
    "description": "ALWAYS validate configuration at startup"
  }
]

### Standard Environment Variables

- `study_DATA_DIR`: Data directory (default: "data/")
- `study_MODEL_DIR`: Model directory (default: "models/")
- `LOG_LEVEL`: Logging level (default: "INFO")

@version "3.0.0"
@last_updated "2025-01-15"
