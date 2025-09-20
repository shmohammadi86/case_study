---
trigger: always_on
description: "Python data science tools and libraries including NumPy, pandas, matplotlib, and scientific computing best practices"
globs: ["**/*.py", "**/*.ipynb", "notebooks/**/*", "scripts/**/*"]
---

# Python Data Science Tools

@context {
  "type": "data_science_tools",
  "primary_libraries": ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"],
  "domain": "scientific_computing",
  "notebook_support": "jupyter"
}

## NumPy Best Practices

@numpy_rules [
  {
    "id": "vectorize_operations",
    "severity": "error",
    "description": "ALWAYS use vectorized operations instead of Python loops"
  },
  {
    "id": "appropriate_dtypes",
    "description": "Use appropriate dtypes for memory efficiency"
  },
  {
    "id": "broadcasting_rules",
    "description": "Understand and leverage NumPy broadcasting"
  },
  {
    "id": "axis_parameters",
    "description": "Use axis parameters correctly for multi-dimensional operations"
  },
  {
    "id": "memory_layout",
    "description": "Consider memory layout (C vs Fortran order) for performance"
  }
]

## Pandas Best Practices

@pandas_rules [
  {
    "id": "method_chaining",
    "description": "Use method chaining for readable data transformations"
  },
  {
    "id": "vectorized_operations",
    "severity": "error",
    "description": "Prefer vectorized operations over apply() when possible"
  },
  {
    "id": "categorical_data",
    "description": "Use categorical data type for repeated string values"
  },
  {
    "id": "memory_optimization",
    "description": "Optimize memory usage with appropriate dtypes"
  },
  {
    "id": "query_method",
    "description": "Use .query() method for complex filtering"
  }
]

### Pandas Performance Guidelines

- Use `.loc[]` and `.iloc[]` for explicit indexing
- Avoid chained assignment (`df['col'][0] = value`)
- Use `.copy()` when modifying DataFrames to avoid warnings
- Leverage `.pipe()` for custom operations in method chains
- Use `.assign()` for adding multiple columns

## Matplotlib Best Practices

@matplotlib_rules [
  {
    "id": "object_oriented_api",
    "description": "Use object-oriented API (fig, ax) instead of pyplot"
  },
  {
    "id": "consistent_styling",
    "description": "Use consistent color schemes and styling"
  },
  {
    "id": "proper_labeling",
    "severity": "error",
    "description": "ALWAYS include axis labels, titles, and legends"
  },
  {
    "id": "figure_size",
    "description": "Set appropriate figure sizes for target medium"
  },
  {
    "id": "color_accessibility",
    "description": "Use colorblind-friendly color palettes"
  }
]

### Plotting Guidelines

- Use `plt.subplots()` for creating figures and axes
- Set figure DPI appropriately for output medium
- Use `tight_layout()` or `constrained_layout` for spacing
- Save figures in appropriate formats (PNG for web, PDF for print)
- Close figures explicitly to free memory

## SciPy Best Practices

@scipy_rules [
  {
    "id": "specialized_functions",
    "description": "Use specialized SciPy functions over generic implementations"
  },
  {
    "id": "sparse_matrices",
    "description": "Use sparse matrices for large, sparse data"
  },
  {
    "id": "optimization_methods",
    "description": "Choose appropriate optimization methods for problem type"
  },
  {
    "id": "statistical_tests",
    "description": "Validate assumptions before applying statistical tests"
  }
]

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
    "id": "data_leakage",
    "severity": "error",
    "description": "Prevent data leakage by fitting preprocessors only on training data"
  },
  {
    "id": "random_state",
    "description": "Set random_state for reproducible results"
  },
  {
    "id": "feature_scaling",
    "description": "Scale features appropriately for distance-based algorithms"
  }
]

## Jupyter Notebook Best Practices

@notebook_rules [
  {
    "id": "cell_organization",
    "description": "Organize code into logical, small cells"
  },
  {
    "id": "markdown_documentation",
    "description": "Use markdown cells to explain analysis steps"
  },
  {
    "id": "reproducible_execution",
    "description": "Ensure notebooks run completely from top to bottom"
  },
  {
    "id": "clear_outputs",
    "description": "Clear outputs before committing to version control"
  },
  {
    "id": "modular_functions",
    "description": "Extract reusable code into functions or modules"
  }
]

## Data Handling

@data_handling_rules [
  {
    "id": "missing_data_strategy",
    "severity": "error",
    "description": "Have explicit strategy for handling missing data"
  },
  {
    "id": "data_validation",
    "severity": "error",
    "description": "Validate data integrity and assumptions"
  },
  {
    "id": "data_versioning",
    "description": "Version datasets and track data lineage"
  },
  {
    "id": "memory_efficient_loading",
    "description": "Use chunking for large datasets that don't fit in memory"
  }
]

## Performance Optimization

@performance_rules [
  {
    "id": "profile_before_optimize",
    "description": "Profile code to identify actual bottlenecks"
  },
  {
    "id": "vectorization_first",
    "description": "Try vectorization before moving to Numba or Cython"
  },
  {
    "id": "appropriate_data_structures",
    "description": "Choose appropriate data structures for the task"
  },
  {
    "id": "lazy_evaluation",
    "description": "Use lazy evaluation when working with large datasets"
  }
]

## Scientific Computing Standards

@scientific_standards [
  {
    "id": "reproducible_research",
    "severity": "error",
    "description": "Ensure all analysis is reproducible"
  },
  {
    "id": "version_tracking",
    "description": "Track versions of all dependencies and data"
  },
  {
    "id": "statistical_rigor",
    "description": "Apply appropriate statistical methods and validate assumptions"
  },
  {
    "id": "clear_methodology",
    "description": "Document methodology and assumptions clearly"
  }
]

@version "1.0.0"
@last_updated "2025-01-14"
