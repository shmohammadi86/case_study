---
trigger: always_on
description: "Python best practices including coding standards, type hints, error handling, and performance optimization"
globs: ["**/*.py", "**/*.pyi"]
---

# Python Best Practices

@context {
  "type": "coding_standards",
  "language": "python",
  "version": "3.12+",
  "style_guide": "PEP_8_extended"
}

## Code Style and Formatting

@style_rules [
  {
    "id": "pep8_compliance",
    "severity": "error",
    "description": "Follow PEP 8 style guide for Python code"
  },
  {
    "id": "line_length",
    "severity": "warning",
    "description": "Keep line lengths reasonable (88 characters with Black)"
  },
  {
    "id": "meaningful_names",
    "severity": "error",
    "description": "Use meaningful variable and function names"
  },
  {
    "id": "no_magic_numbers",
    "severity": "warning",
    "description": "Avoid magic numbers and hardcoded values"
  }
]

## Type Hints

@type_hints_rules [
  {
    "id": "mandatory_type_hints",
    "severity": "error",
    "description": "ALL public functions MUST have type hints"
  },
  {
    "id": "return_type_annotation",
    "severity": "error",
    "description": "ALL functions MUST have return type annotations"
  },
  {
    "id": "generic_types",
    "description": "Use generic types for collections and protocols"
  },
  {
    "id": "optional_types",
    "description": "Use Optional[T] or T | None for optional parameters"
  }
]

## Error Handling

@error_handling_rules [
  {
    "id": "specific_exceptions",
    "severity": "error",
    "description": "Use specific exception types, not generic Exception"
  },
  {
    "id": "informative_messages",
    "severity": "error",
    "description": "Provide informative error messages with context"
  },
  {
    "id": "no_silent_failures",
    "severity": "error",
    "description": "NEVER silently catch and ignore exceptions"
  },
  {
    "id": "exception_chaining",
    "description": "Use exception chaining with 'raise ... from ...' when appropriate"
  }
]

## Performance

@performance_rules [
  {
    "id": "avoid_premature_optimization",
    "description": "Write clear code first, optimize when needed"
  },
  {
    "id": "list_comprehensions",
    "description": "Prefer list comprehensions over loops when appropriate"
  },
  {
    "id": "generator_expressions",
    "description": "Use generator expressions for memory efficiency"
  },
  {
    "id": "string_operations",
    "description": "Use str.join() for multiple string concatenations"
  }
]

## Naming Conventions

@naming_conventions {
  "variables": "snake_case",
  "functions": "snake_case",
  "classes": "PascalCase",
  "constants": "UPPER_SNAKE_CASE",
  "modules": "snake_case",
  "packages": "snake_case"
}

## Function Design

@function_rules [
  {
    "id": "single_responsibility",
    "severity": "error",
    "description": "Each function MUST have a single, clear responsibility"
  },
  {
    "id": "pure_functions",
    "description": "Prefer pure functions with no side effects"
  },
  {
    "id": "parameter_validation",
    "severity": "error",
    "description": "Validate function parameters at entry points"
  },
  {
    "id": "return_consistency",
    "description": "Return consistent types from functions"
  }
]

## Class Design

@class_rules [
  {
    "id": "composition_over_inheritance",
    "description": "Prefer composition over inheritance"
  },
  {
    "id": "minimal_interfaces",
    "description": "Keep class interfaces minimal and focused"
  },
  {
    "id": "immutable_when_possible",
    "description": "Make classes immutable when possible"
  },
  {
    "id": "proper_str_repr",
    "description": "Implement __str__ and __repr__ for debugging"
  }
]

## Import Organization

@import_rules [
  {
    "id": "import_order",
    "severity": "error",
    "description": "Follow standard import order: stdlib, third-party, local"
  },
  {
    "id": "explicit_imports",
    "severity": "error",
    "description": "Use explicit imports, avoid 'import *'"
  },
  {
    "id": "absolute_imports",
    "severity": "error",
    "description": "Use absolute imports for project modules"
  }
]

## Constants and Configuration

@constants_rules [
  {
    "id": "module_level_constants",
    "description": "Define constants at module level in UPPER_SNAKE_CASE"
  },
  {
    "id": "enum_for_choices",
    "description": "Use Enum for related constants and choices"
  },
  {
    "id": "config_classes",
    "description": "Use dataclasses or Pydantic for configuration"
  }
]

@version "1.0.0"
@last_updated "2025-01-14"
