---
trigger: manual
description:
globs:
---"),
        section_content,
        mo.md("---\n\n**Happy coding! ")
    ])
```

## UI/UX BEST PRACTICES

### Modern Styling
- **Use gradient backgrounds for visual appeal**
- **Consistent color coding (blue=info, green=success, yellow=warning, red=error)**
- **Responsive grid layouts for multiple columns**

```python
# Modern card-style layout
mo.md(r"""
<div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid #1565c0;">
    <h4 style="margin-top: 0; color: #1565c0;"> Design Principle</h4>
    <p>Detailed explanation with proper contrast and readability.</p>
</div>
""")

# Responsive grid for concepts
mo.md(r"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0;">
    <div style="background: linear-gradient(135deg, #e8f5e8, #c8e6c9); padding: 1rem; border-radius: 8px;">
        <h4 style="margin-top: 0; color: #2e7d32;"> Concept 1</h4>
        <p style="font-size: 0.9rem;">Explanation</p>
    </div>
    <div style="background: linear-gradient(135deg, #fff3e0, #ffcc02); padding: 1rem; border-radius: 8px;">
        <h4 style="margin-top: 0; color: #f57c00;"> Concept 2</h4>
        <p style="font-size: 0.9rem;">Explanation</p>
    </div>
</div>
""")
```

### Accessibility Features
- **Semantic HTML structure**
- **Proper color contrast ratios** 
- **Keyboard navigation support**
- **Screen reader friendly content**

```python
# Accessible controls with proper labels
parameter_control = mo.ui.slider(
    start=0.05, stop=0.5, step=0.05, value=0.1,
    label=" Edge Density (affects network connectivity)",
    show_value=True
)

debug_mode = mo.ui.checkbox(
    value=False,
    label=" Debug Mode (show detailed logging output)"
)
```

## PERFORMANCE OPTIMIZATION

### Lazy Loading and Caching
- **Generate expensive content only when needed**
- **Cache computation results**
- **Use conditional rendering for performance**

```python
@app.cell
def expensive_computation(mo, should_compute, cached_results):
    """Only compute when explicitly requested."""
    if not should_compute.value:
        return mo.md("Click 'Run Computation' to generate results.")
    
    # Check cache first
    if 'results' in cached_results:
        results = cached_results['results']
    else:
        # Expensive operation
        results = complex_analysis()
        cached_results['results'] = results
    
    return mo.md(f"**Results**: {results}")
```

### Memory Management
- **Clear large variables when not needed**
- **Use generators for large datasets**
- **Monitor memory usage in debug mode**

## DEBUGGING AND TROUBLESHOOTING

### Common Issues and Solutions

#### MultipleDefinitionError
```python
# CAUSE: Same variable name in multiple cells
@app.cell
def cell1():
    data = [1, 2, 3]  # Variable 'data' defined
    
@app.cell  
def cell2():
    data = [4, 5, 6]  # ERROR: 'data' redefined

# SOLUTION: Unique variable names
@app.cell
def cell1():
    input_data = [1, 2, 3]
    return (input_data,)
    
@app.cell
def cell2(input_data):
    processed_data = transform(input_data)
    return (processed_data,)
```

#### Cell Dependency Issues
```python
# CAUSE: Implicit dependencies
@app.cell
def bad_cell():
    result = some_function(data)  # 'data' not passed as parameter

# SOLUTION: Explicit dependencies
@app.cell
def good_cell(data):  # Explicit dependency
    result = some_function(data)
    return result
```

#### UI Control Issues
```python
# CAUSE: Control not returned from function
@app.cell
def bad_controls():
    slider = mo.ui.slider(0, 100, 1, 50)
    # slider not returned - cannot be used elsewhere

# SOLUTION: Return controls
@app.cell
def good_controls(mo):
    slider = mo.ui.slider(0, 100, 1, 50)
    return (slider,)  # Return as tuple
```

### Debug Mode Implementation
```python
@app.cell
def debug_info(mo, debug_mode, import_status):
    """Show debug information when enabled."""
    if not debug_mode.value:
        return None
        
    debug_content = []
    debug_content.append("## Debug Information")
    debug_content.append(f"**Import Status**: {import_status}")
    debug_content.append(f"**Memory Usage**: {get_memory_info()}")
    
    return mo.md("\n".join(debug_content))
```

## INTEGRATION WITH study ARCHITECTURE

### 4-Layer Architecture Context
- **Always show current layer position**
- **Link to related tutorials**
- **Consistent architecture visualization**

```python
def show_architecture_context(current_layer="data"):
    """Visual representation of 4-layer architecture."""
    layers = {
        "data": "#007bff",
        "module": "#6c757d", 
        "model": "#6c757d",
        "executor": "#6c757d"
    }
    layers[current_layer] = "#007bff"  # Highlight current
    
    return mo.md(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
        <div style="background: {layers['data']}; color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong> Data Layer</strong>
        </div>
        <div style="background: {layers['module']}; color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong> Module Layer</strong>
        </div>
        <div style="background: {layers['model']}; color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong> Model Layer</strong>
        </div>
        <div style="background: {layers['executor']}; color: white; padding: 0.75rem; border-radius: 8px; text-align: center;">
            <strong> Executor Layer</strong>
        </div>
    </div>
    """)
```

## FINAL CHECKLIST

Before deploying any Marimo tutorial:

- [ ] All cells are functions that return `mo` objects
- [ ] No variable name conflicts across cells  
- [ ] Dependencies passed explicitly between cells
- [ ] Interactive controls returned from functions
- [ ] Error handling with graceful fallbacks
- [ ] Progressive disclosure with conditional rendering
- [ ] Accessibility features implemented
- [ ] Debug mode available
- [ ] Architecture context shown
- [ ] Performance optimizations applied
- [ ] Tested with `nox -s marimo-run` in headless mode

**Remember**: These patterns are production-tested and ensure consistent, high-quality tutorial development across the entire study project. Pitfalls to Avoid

1. **Multiple Definition Errors**: Don't reuse variable names across cells
2. **Return Statement Issues**: Avoid multiple returns in conditionals  
3. **Import Conflicts**: Don't import same modules in multiple cells
4. **Missing Dependencies**: Always handle import/file failures gracefully
{{ ... }}
6. **Display Issues**: Ensure cells produce visible output
7. **Variable Scoping**: Don't use underscores for shared variables

## Tutorial Structure Template

### Recommended Cell Order
1. **Imports Cell**: Central dependency management
2. **Introduction Cell**: Overview, objectives, Mermaid diagrams
3. **Core Concept Cells**: Main tutorial content with working demos
4. **Enhanced Feature Cells**: Advanced capabilities with fallbacks
5. **Integration Cells**: Show framework integration patterns
6. **Production Pattern Cells**: Enterprise-ready examples
7. **Summary Cell**: Achievements, key learnings, next steps

### Robust Cell Template
```python
@app.cell
def section_name(dependencies, mo):
    """Brief description of cell purpose."""
    _title = mo.md("## Section Title").center()
    
    try:
        if conditions_met:
            _content = mo.callout(
                mo.md("Success content with working demo"),
                kind="success"
            )
        else:
            _content = mo.callout(
                mo.md("Educational fallback explanation"),
                kind="warn"
            )
    except Exception as e:
        _content = mo.callout(
            mo.md(f"Error handling: {str(e)}"),
            kind="danger"
        )
    
    # Single final expression for display
    mo.vstack([_title, _content])
```

## Professional Patterns

### Statistics Display
```python
_stats = [
    mo.stat(value=f"{count:,}", label="Items", caption="Total count"),
    mo.stat(value=f"{size:.1f} MB", label="Size", caption="Data size")
]
mo.hstack(_stats, justify="space-around")
```

### Educational Content Structure
- Lead with clear objectives
- Provide working examples when possible
- Explain capabilities even when demonstrating fallbacks
- Include practical next steps and resources
- Use consistent visual hierarchy throughout

### Content Organization
- Group related functionality in logical sections
- Use descriptive cell names that explain purpose
- Maintain consistent parameter passing patterns
- Create reusable patterns for common scenarios
- Document complex logic with clear comments

These rules ensure creation of professional, robust, and maintainable Marimo notebooks that provide excellent user experience regardless of environment limitations.
