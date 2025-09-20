---
trigger: always_on
description: "Documentation standards including docstrings, API docs, README files, and technical writing guidelines"
globs: ["**/*.py", "**/*.md", "docs/**/*", "README*", "CONTRIBUTING*", "CHANGELOG*"]
---

# Documentation Standards

@context {
  "type": "documentation",
  "docstring_format": "numpy",
  "api_docs": "sphinx",
  "markdown_flavor": "github"
}

## Docstring Requirements

@docstring_rules [
  {
    "id": "required_for_public_apis",
    "severity": "error",
    "description": "ALL public functions and classes MUST have docstrings"
  },
  {
    "id": "numpy_format",
    "severity": "error",
    "description": "Use NumPy docstring format consistently"
  },
  {
    "id": "complete_parameters",
    "severity": "error",
    "description": "Document ALL parameters with types and descriptions"
  },
  {
    "id": "return_documentation",
    "severity": "error",
    "description": "Document return values with types and descriptions"
  }
]

## Docstring Structure

@docstring_structure {
  "summary": "One-line summary ending with period",
  "description": "Detailed description if needed",
  "parameters": "Document all parameters with types",
  "returns": "Document return value with type",
  "raises": "Document exceptions that may be raised",
  "examples": "Provide usage examples for complex functions"
}

## API Documentation

@api_docs_rules [
  {
    "id": "sphinx_autodoc",
    "description": "Use Sphinx autodoc for API documentation generation"
  },
  {
    "id": "type_annotations",
    "description": "Include type annotations in generated docs"
  },
  {
    "id": "cross_references",
    "description": "Use Sphinx cross-references between modules"
  }
]

## README Requirements

@readme_rules [
  {
    "id": "project_description",
    "severity": "error",
    "description": "Clearly describe what the project does"
  },
  {
    "id": "installation_instructions",
    "severity": "error",
    "description": "Provide clear installation instructions"
  },
  {
    "id": "usage_examples",
    "severity": "error",
    "description": "Include basic usage examples"
  },
  {
    "id": "contribution_guidelines",
    "description": "Include contribution guidelines"
  }
]

### README Template Structure

1. **Project Title and Description**
2. **Features/Key Benefits**
3. **Installation Instructions**
4. **Quick Start/Usage Examples**
5. **Documentation Links**
6. **Contributing Guidelines**
7. **License Information**

## Code Comments

@comment_rules [
  {
    "id": "explain_why_not_what",
    "severity": "error",
    "description": "Comments should explain WHY, not WHAT"
  },
  {
    "id": "complex_logic_explanation",
    "description": "Explain complex algorithms and business logic"
  },
  {
    "id": "no_obvious_comments",
    "description": "Avoid comments that state the obvious"
  },
  {
    "id": "update_with_code",
    "severity": "error",
    "description": "Keep comments in sync with code changes"
  }
]

## Jupyter Notebooks

@notebook_rules [
  {
    "id": "clear_structure",
    "description": "Use clear section headers and markdown explanations"
  },
  {
    "id": "reproducible_results",
    "description": "Ensure notebooks run completely and reproducibly"
  },
  {
    "id": "clean_outputs",
    "description": "Clear outputs before committing notebooks"
  },
  {
    "id": "minimal_notebooks",
    "description": "Keep notebooks focused and minimal"
  }
]

## Changelog Maintenance

@changelog_rules [
  {
    "id": "keep_changelog",
    "description": "Maintain CHANGELOG.md following Keep a Changelog format"
  },
  {
    "id": "semantic_versioning",
    "description": "Use semantic versioning for releases"
  },
  {
    "id": "categorize_changes",
    "description": "Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security"
  }
]

## Sphinx Configuration

@sphinx_rules [
  {
    "id": "theme_consistency",
    "description": "Use consistent theme and styling"
  },
  {
    "id": "extension_usage",
    "description": "Use relevant Sphinx extensions for enhanced documentation"
  },
  {
    "id": "build_automation",
    "description": "Automate documentation builds in CI/CD"
  }
]

### Recommended Sphinx Extensions

- `sphinx.ext.autodoc`: Auto-generate API docs
- `sphinx.ext.napoleon`: Parse NumPy/Google style docstrings
- `sphinx.ext.viewcode`: Include source code links
- `sphinx.ext.intersphinx`: Link to other project docs

## Documentation Testing

@docs_testing_rules [
  {
    "id": "doctest_examples",
    "description": "Use doctest for testable code examples"
  },
  {
    "id": "link_checking",
    "description": "Regularly check for broken links"
  },
  {
    "id": "build_validation",
    "description": "Validate documentation builds without warnings"
  }
]

## Multilingual Support

@i18n_rules [
  {
    "id": "internationalization_ready",
    "description": "Structure documentation to support internationalization"
  },
  {
    "id": "consistent_terminology",
    "description": "Use consistent terminology throughout documentation"
  }
]

@version "1.0.0"
@last_updated "2025-01-14"
