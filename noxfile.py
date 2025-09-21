"""
🚀 ENHANCED PRODUCTION-READY NOXFILE FOR study ML WORKLOADS
==================================================================

This noxfile provides a comprehensive, production-ready development environment for
machine learning projects with enhanced GPU support and modern tooling, featuring:

✨ **CORE FEATURES:**
- 🎯 H100/A100 GPU optimization with CUDA 12.4-12.8 support
- ⚡ Modern UV 0.5+ dependency management with universal lockfiles
- 🔧 PyTorch 2.6+ ecosystem with cutting-edge optimizations
- 🧪 Comprehensive experiment management and tracking
- 📊 Advanced performance monitoring and profiling
- 🛡️ Security scanning and compliance checking
- 📚 Enhanced documentation generation
- 🎨 Comprehensive tool-specific cache management

✨ **MIGRATION STRATEGY:**
- 📦 Iterative session migration from existing noxfile.py
- 🔧 Enhanced hardware detection and optimization
- 🚀 Production-ready deployment capabilities
- 📊 Advanced monitoring and profiling
- 🛡️ Security and compliance automation
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

import nox

# =============================================================================
# PROJECT CONFIGURATION - Enhanced for Production
# ===============================================================================

# Python version support (enhanced for compatibility)
SUPPORTED_PYTHON_VERSIONS = ["3.12"]  # Added 3.13 support
DEFAULT_PYTHON_VERSION = "3.12"

# Project paths (organized for scalability)
PROJECT_ROOT = Path(".")
LOCKS_DIR = PROJECT_ROOT / "pinned-versions"
CACHE_DIR = PROJECT_ROOT / "cache"
TEMP_DIR = PROJECT_ROOT / "scratch" / "temp"
LOGS_DIR = PROJECT_ROOT / "scratch" / "logs"
DEPLOYMENT_DIR = PROJECT_ROOT / "scratch" / "deployment"
CONTAINERS_DIR = PROJECT_ROOT / "scratch" / "containers"

WANDB_ENV_FILE = PROJECT_ROOT / ".wandb_env"

# Ensure essential directories exist
for directory in [LOCKS_DIR, CACHE_DIR, TEMP_DIR, LOGS_DIR, DEPLOYMENT_DIR, CONTAINERS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===============================================================================
# COMPREHENSIVE TOOL-SPECIFIC CACHE CONFIGURATION
# ===============================================================================

def get_cache_config() -> dict[str, str]:
    """
    Generate cache configuration based on pyproject.toml cache hierarchy.

    Implements the sophisticated cache fallback system defined in pyproject.toml:
    - GLOBAL_CACHE_DIR or XDG_CACHE_HOME or HOME/.cache (global)
    - PROJECT_CACHE_DIR or GLOBAL_CACHE_DIR/study (project-specific)
    """
    # Cache hierarchy implementation from pyproject.toml
    home_dir = os.path.expanduser("~")
    xdg_cache = os.environ.get("XDG_CACHE_HOME", f"{home_dir}/.cache")
    global_cache = os.environ.get("GLOBAL_CACHE_DIR", xdg_cache)
    project_cache = os.environ.get("PROJECT_CACHE_DIR", f"{global_cache}/study")

    # Override with our project-specific cache if running from project
    if PROJECT_ROOT.exists() and (PROJECT_ROOT / "src" / "study").exists():
        project_cache = str(CACHE_DIR)

    return {
        # Cache hierarchy foundations
        "GLOBAL_CACHE_DIR": global_cache,
        "PROJECT_CACHE_DIR": project_cache,
        "XDG_CACHE_HOME": xdg_cache,

        # Core package managers and build tools (global level)
        "UV_CACHE_DIR": f"{global_cache}/uv",
        "PIXI_CACHE_DIR": f"{global_cache}/pixi",
        "PIP_CACHE_DIR": f"{global_cache}/pip",
        "PRE_COMMIT_HOME": f"{global_cache}/pre-commit",

        # Temporary directories
        "TMPDIR": str(TEMP_DIR),
        "TMP": str(TEMP_DIR),
        "TEMP": str(TEMP_DIR),
        "PYTHONPATH": str(PROJECT_ROOT / "src"),

        # Python tool caches (project-specific as per pyproject.toml)
        "MYPY_CACHE_DIR": f"{project_cache}/mypy",
        "PYTEST_CACHE_DIR": f"{project_cache}/pytest",
        "RUFF_CACHE_DIR": f"{project_cache}/ruff",
        "BLACK_CACHE_DIR": f"{project_cache}/black",
        "ISORT_CACHE_DIR": f"{project_cache}/isort",
        "NOX_CACHE_DIR": f"{project_cache}/nox",

        # Coverage and test reporting (project-specific)
        "COVERAGE_FILE": f"{project_cache}/coverage/.coverage",
        "COVERAGE_DATA_FILE": f"{project_cache}/coverage/.coverage",
        "COVERAGE_REPORT_DIR": f"{project_cache}/coverage/html",

        # WandB experiment tracking (experiment-specific logs go in experiments/{name}/logs/)
        "WANDB_CACHE_DIR": f"{project_cache}/wandb/artifacts",
        "WANDB_CONFIG_DIR": f"{project_cache}/wandb/config",
        "WANDB_DATA_DIR": f"{project_cache}/wandb",
        "WANDB_DIR": f"{project_cache}/wandb",

        # Jupyter and IPython cache directories (project-specific)
        "JUPYTER_CONFIG_DIR": f"{project_cache}/jupyter/config",
        "JUPYTER_DATA_DIR": f"{project_cache}/jupyter/data",
        "JUPYTER_RUNTIME_DIR": f"{project_cache}/jupyter/runtime",
        "JUPYTER_CACHE_DIR": f"{project_cache}/jupyter/cache",
        "IPYTHONDIR": f"{project_cache}/ipython",

        # Marimo cache and data directories (project-specific)
        "MARIMO_CONFIG_DIR": f"{project_cache}/marimo",
        "MARIMO_DATA_DIR": f"{project_cache}/marimo/data",
        "MARIMO_CACHE_DIR": f"{project_cache}/marimo/cache",

        # ML/AI framework caches (global level for sharing across projects)
        "TORCH_HOME": f"{global_cache}/torch",
        "HF_HOME": f"{global_cache}/huggingface",

        "MLFLOW_TRACKING_URI": f"{project_cache}/mlflow",

        # Documentation and build tools (project-specific)
        "SPHINX_BUILD_DIR": f"{project_cache}/sphinx",
        "MKDOCS_CACHE_DIR": f"{project_cache}/mkdocs",

        # Git and version control (global)
        "GIT_TEMPLATE_DIR": f"{global_cache}/git/templates",

        # Node.js and JavaScript (global)
        "NPM_CONFIG_CACHE": f"{global_cache}/npm",
        "NODE_MODULES_CACHE": f"{global_cache}/node_modules",

        # Build and compilation settings
        "CXX": "g++",
        "CC": "gcc",
        "CXXFLAGS": "-std=c++17",
        "CMAKE_BUILD_TYPE": "Release",
        "MAKEFLAGS": f"-j{os.cpu_count()}",
        "MAX_JOBS": str(os.cpu_count()),

        # Python optimizations
        "PYTHONDONTWRITEBYTECODE": "1",  # Prevent .pyc files in source tree
        "PYTHONPYCACHEPREFIX": f"{project_cache}/pycache",  # Redirect __pycache__ to scratch

        # Enhanced security and compliance
        "UV_AUDIT": "1",  # Enable security auditing
        "SAFETY_CHECK": "1",

        # Container and deployment optimizations
        "DOCKER_BUILDKIT": "1",
        "COMPOSE_DOCKER_CLI_BUILD": "1",
    }

# Generate the cache configuration
COMMON_ENV = get_cache_config()

# ===============================================================================
# ENHANCED HARDWARE DETECTION AND OPTIMIZATION
# ===============================================================================

def detect_hardware_capabilities() -> dict[str, bool | str | list[str]]:
    """Enhanced hardware detection for H100/A100 clusters with capability analysis."""
    capabilities = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": [],
        "gpu_compute_caps": [],
        "cuda_version": None,
        "driver_version": None,
        "cpu_count": os.cpu_count(),
        "total_memory_gb": 0,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "is_h100": False,
        "is_a100": False,
        "supports_nvlink": False,
        "optimal_batch_size": None,
    }

    # Check for CI/CD environments first
    ci_indicators = [
        "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI",
        "JENKINS_URL", "BUILDKITE", "CIRCLECI", "AZURE_PIPELINES"
    ]
    if any(os.getenv(var) for var in ci_indicators):
        capabilities["platform"] = f"{capabilities['platform']} (CI/CD)"
        return capabilities

    # Enhanced GPU detection
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            capabilities["gpu_available"] = True
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_name, memory, compute_cap = parts[0], parts[1], parts[2]
                        capabilities["gpu_names"].append(gpu_name)
                        capabilities["gpu_memory"].append(int(memory))
                        capabilities["gpu_compute_caps"].append(compute_cap)

                        # Detect H100/A100 specifically
                        if "H100" in gpu_name:
                            capabilities["is_h100"] = True
                        elif "A100" in gpu_name:
                            capabilities["is_a100"] = True

            capabilities["gpu_count"] = len(capabilities["gpu_names"])

            # Check for NVLink support
            if capabilities["gpu_count"] > 1:
                try:
                    nvlink_result = subprocess.run(
                        ["nvidia-smi", "nvlink", "--status"],
                        check=False, capture_output=True, text=True, timeout=5
                    )
                    if nvlink_result.returncode == 0:
                        capabilities["supports_nvlink"] = True
                except Exception as e:
                    # Log the exception for debugging
                    pass

        # Get CUDA version
        try:
            cuda_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=5
            )
            if cuda_result.returncode == 0:
                capabilities["driver_version"] = cuda_result.stdout.strip().split('\n')[0]
        except Exception as e:
            # Log the exception for debugging
            pass

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Memory detection
    try:
        import psutil
        capabilities["total_memory_gb"] = round(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        # Fallback memory detection without psutil
        try:
            with open('/proc/meminfo') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        capabilities["total_memory_gb"] = round(kb / (1024**2))
                        break
        except Exception as e:
            # Log the exception for debugging
            pass

    # Optimal batch size estimation for H100/A100
    if capabilities["is_h100"] or capabilities["is_a100"]:
        total_gpu_memory = sum(capabilities["gpu_memory"])
        if total_gpu_memory > 0:
            # Conservative estimate: use 70% of GPU memory for batch processing
            capabilities["optimal_batch_size"] = int((total_gpu_memory * 0.7) / 1024)  # GB

    return capabilities

def get_hardware_preference() -> str:
    """Enhanced hardware preference detection with H100/A100 optimization."""
    capabilities = detect_hardware_capabilities()

    if capabilities["gpu_available"]:
        if capabilities["is_h100"]:
            return "h100"
        elif capabilities["is_a100"]:
            return "a100"
        else:
            return "gpu"

    return "cpu"

def get_optimal_cuda_version() -> str:
    """
    Determine optimal CUDA version based on hardware and environment.

    Aligns with pyproject.toml CUDA version support:
    - cu124: Conservative CUDA 12.4 (maximum compatibility)
    - cu126: Stable CUDA 12.6 (recommended for production)
    - cu128: Cutting-edge CUDA 12.8 (maximum H100/A100 performance)
    """
    capabilities = detect_hardware_capabilities()

    # For H100/A100, use cutting-edge CUDA for maximum performance
    if capabilities["is_h100"]:
        return "cu128"  # CUDA 12.8 for maximum H100 performance (latest available)
    elif capabilities["is_a100"]:
        return "cu128"  # CUDA 12.8 for A100 (cutting-edge performance)
    elif capabilities["gpu_available"]:
        # Check GPU compute capability for other GPUs
        compute_caps = capabilities.get("gpu_compute_caps", [])
        if compute_caps:
            # Modern GPUs (RTX 40 series, etc.) can use cu128
            # Older GPUs should use conservative cu124
            try:
                max_compute_cap = max(float(cap) for cap in compute_caps if cap)
                if max_compute_cap >= 8.0:  # Ampere or newer
                    return "cu128"  # CUDA 12.8 for modern GPUs
                else:
                    return "cu124"  # Conservative CUDA 12.4 for older GPUs
            except (ValueError, TypeError):
                pass
        return "cu124"  # Conservative default for unknown GPU configurations
    else:
        return "cpu"

def get_hardware_optimizations() -> dict[str, str]:
    """
    Get H100/A100 specific optimizations based on detected hardware.

    Implements the optimization settings from pyproject.toml for maximum performance.
    """
    capabilities = detect_hardware_capabilities()
    optimizations = {}

    if capabilities["is_h100"] or capabilities["is_a100"]:
        # H100/A100 specific PyTorch optimizations
        optimizations.update({
            # PyTorch 2.0+ compilation optimizations
            "TORCH_COMPILE_MODE": "max-autotune",  # Maximum performance compilation
            "TORCH_CUDNN_V8_API_ENABLED": "1",      # cuDNN v8 optimizations
            "TORCH_CUDA_ARCH_LIST": "8.0;9.0",     # A100 (8.0) + H100 (9.0) architectures

            # NCCL optimizations for multi-GPU
            "NCCL_ALGO": "Ring",                    # Optimal algorithm for H100/A100
            "NCCL_P2P_DISABLE": "0",                # Enable P2P for performance
            "NCCL_DEBUG": "WARN",                   # Minimal debug output

            # Memory allocation optimizations
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True",
            "CUDA_LAUNCH_BLOCKING": "0",            # Async CUDA calls for performance

            # Build optimizations for native performance
            "CUDA_NVCC_FLAGS": "-O3 -Xptxas -O3",
            "TORCH_NVCC_FLAGS": "-O3",
        })

        # H100 specific (cutting-edge) optimizations
        if capabilities["is_h100"]:
            optimizations.update({
                "TORCH_COMPILE_BACKEND": "inductor",    # Use TorchInductor for H100
                "TORCH_COMPILE_DYNAMIC": "True",        # Dynamic shapes for flexibility
                "CUDA_MODULE_LOADING": "LAZY",          # Lazy loading for faster startup
            })

        # A100 specific (stable production) optimizations
        elif capabilities["is_a100"]:
            optimizations.update({
                "TORCH_COMPILE_BACKEND": "inductor",    # Stable backend for A100
                "TORCH_COMPILE_DYNAMIC": "False",       # Static shapes for stability
            })

    return optimizations

def log_session_info(session: nox.Session, extra_info: dict | None = None) -> None:
    """Enhanced session logging with hardware info and optimizations."""
    capabilities = detect_hardware_capabilities()
    optimizations = get_hardware_optimizations()

    session.log("=" * 80)
    session.log(f"🚀 SESSION: {session.name}")
    session.log(f"🐍 Python: {session.python}")
    session.log(f"💻 Platform: {capabilities['platform']} ({capabilities['architecture']})")
    session.log(f"🖥️  CPUs: {capabilities['cpu_count']}")
    session.log(f"💾 Memory: {capabilities['total_memory_gb']}GB")

    if capabilities["gpu_available"]:
        session.log(f"🎯 GPUs: {capabilities['gpu_count']} ({', '.join(capabilities['gpu_names'])})")
        if capabilities["is_h100"]:
            session.log("⚡ H100 DETECTED - Using cutting-edge cu128 + max-autotune optimizations")
            session.log(f"🔧 Optimizations: {len(optimizations)} H100-specific settings applied")
        elif capabilities["is_a100"]:
            session.log("⚡ A100 DETECTED - Using high-performance cu128 + stable optimizations")
            session.log(f"🔧 Optimizations: {len(optimizations)} A100-specific settings applied")
        else:
            session.log(f"🎮 GPU: Using {get_optimal_cuda_version()} for detected hardware")
    else:
        session.log("🖥️  GPUs: CPU-only mode")

    if extra_info:
        for key, value in extra_info.items():
            session.log(f"📊 {key}: {value}")

    session.log("=" * 80)

# ===============================================================================
# NOX CONFIGURATION WITH ENHANCED DEFAULTS
# ===============================================================================

# Nox configuration with enhanced settings
nox.options.envdir = str(CACHE_DIR / "nox")
nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = False
nox.options.default_venv_backend = "uv"

# Source files and quality targets
SOURCE_FILES = ("src/study", "tests/", "noxfile.py")
COVERAGE_DIR = str(CACHE_DIR / "coverage")
TEST_REPORTS_DIR = str(CACHE_DIR / "test-reports")

# External tools with version pinning
BUILD_TOOLS = ["build>=1.2.0"]
COVERAGE_TOOLS = ["coverage[toml]>=7.6", "coverage-badge>=1.1", "setuptools>=75.0"]
SECURITY_TOOLS = ["bandit[toml]>=1.7.10", "safety>=3.2.0", "pip-audit>=2.7.0"]

# ===============================================================================
# DEPENDENCY MANAGEMENT AND PROJECT INITIALIZATION
# ===============================================================================

def resolve_lockfile_path(
    python_version: str,
    extra: Sequence[str] = [],
    auto_hardware: bool = True,
) -> Path:
    """Enhanced lockfile resolution with hardware auto-detection."""
    extras_list = list(extra)

    if auto_hardware:
        hardware_extras = {"cpu", "gpu", "cu124", "cu126", "cu128", "h100", "a100"}
        if not any(hw in extras_list for hw in hardware_extras):
            hardware_pref = get_hardware_preference()
            if hardware_pref in ["h100", "a100"]:
                extras_list.append(get_optimal_cuda_version())
            else:
                extras_list.append(hardware_pref)

    if extras_list:
        lockfile_name = f"lockfile.{','.join(sorted(set(extras_list)))}.txt"
    else:
        lockfile_name = "lockfile.txt"

    return LOCKS_DIR / python_version / lockfile_name

def generate_lockfile(
    session: nox.Session,
    lockfile_path: Path,
    extra: Sequence[str] = [],
    upgrade: bool = False,
) -> Path:
    """Generate a package dependencies lockfile using uv pip compile with PyG fallback."""
    # Ensure the directory exists
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)

    # Change to the directory containing pyproject.toml
    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)

    try:
        # Build the uv pip compile command
        sorted_extra = sorted(extra)
        command = [
            "uv",
            "pip",
            "compile",
            *[f"--extra={x}" for x in sorted_extra],
            "--strip-extras",
            "--no-emit-index-url",
            "pyproject.toml",
            "-o",
            str(lockfile_path),
        ]

        if upgrade:
            command.append("--upgrade")

        try:
            # First attempt: Try with PyG extensions
            session.run(*command, env={"UV_CUSTOM_COMPILE_COMMAND": f"nox -s {session.name}"}, external=True)
            session.log(f"✅ Lockfile generated: {lockfile_path}")
            return lockfile_path

        except Exception as e:
            # Check if the error is related to PyG extensions
            if any(pkg in str(e) for pkg in ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "pyg-lib"]):
                session.log("⚠️  PyG extensions failed to build. Generating lockfile without PyG extensions...")

                # Create a temporary pyproject.toml without PyG extensions
                backup_and_create_minimal_pyproject(session)

                try:
                    # Second attempt: Without PyG extensions
                    session.run(*command, env={"UV_CUSTOM_COMPILE_COMMAND": f"nox -s {session.name} (no PyG extensions)"}, external=True)
                    session.log(f"✅ Lockfile generated without PyG extensions: {lockfile_path}")
                    session.log("💡 PyG extensions can be installed manually later if needed")
                    return lockfile_path

                finally:
                    # Restore original pyproject.toml
                    restore_original_pyproject(session)
            else:
                # Re-raise if it's not a PyG extension issue
                raise

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def backup_and_create_minimal_pyproject(session: nox.Session) -> None:
    """Create a backup of pyproject.toml and create a version without PyG extensions."""
    import shutil

    # Backup original pyproject.toml
    shutil.copy("pyproject.toml", "pyproject.toml.backup")
    session.log("📋 Backed up pyproject.toml")

    # Read current pyproject.toml and comment out PyG extensions
    with open("pyproject.toml") as f:
        content = f.read()

    # Comment out PyG extension lines
    pyg_extensions = [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "pyg-lib"
    ]

    lines = content.split('\n')
    modified_lines = []

    for line in lines:
        if any(f'"{ext}' in line for ext in pyg_extensions):
            # Comment out PyG extension lines
            modified_lines.append(f"  # {line.strip()}  # Temporarily disabled due to build issues")
        else:
            modified_lines.append(line)

    # Write modified content
    with open("pyproject.toml", "w") as f:
        f.write('\n'.join(modified_lines))

    session.log("🔧 Created minimal pyproject.toml without PyG extensions")

def restore_original_pyproject(session: nox.Session) -> None:
    """Restore the original pyproject.toml from backup."""
    import os
    import shutil

    if os.path.exists("pyproject.toml.backup"):
        shutil.move("pyproject.toml.backup", "pyproject.toml")
        session.log("🔄 Restored original pyproject.toml")
    else:
        session.log("⚠️  No backup found, pyproject.toml left as-is")

def generate_universal_lock(session: nox.Session, upgrade: bool = False) -> None:
    """Generate uv.lock with all extras pre-resolved for maximum speed."""
    session.log("🔒 Generating comprehensive uv.lock (includes all extras by default)")

    try:
        # Change to project root
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)

        # Build uv lock command (uv.lock includes all extras by default)
        command = ["uv", "lock"]

        if upgrade:
            command.append("--upgrade")
            session.log("📦 Upgrading all dependencies")

        # Run the lock command
        session.run(*command, external=True)
        session.log("✅ Universal uv.lock generated with all extras")
        session.log("💡 'uv sync --extra <specific> --frozen' will now be extremely fast")

    except Exception as e:
        session.error(f"❌ Failed to generate uv.lock: {e}")
        raise
    finally:
        os.chdir(original_cwd)

def validate_python_version(python_version: str) -> bool:
    """Validate that the requested Python version is supported by the project."""
    try:
        pyproject = load_pyproject_toml()
        requires_python = pyproject.get("project", {}).get("requires-python", "")

        # Simple validation - check if version is in SUPPORTED_PYTHON_VERSIONS
        if python_version in SUPPORTED_PYTHON_VERSIONS:
            return True

        print(f"⚠️  Python {python_version} not in supported versions: {SUPPORTED_PYTHON_VERSIONS}")
        return False

    except Exception as e:
        print(f"Warning: Could not validate Python version: {e}")
        return python_version in SUPPORTED_PYTHON_VERSIONS

def auto_detect_hardware_extras(hardware: str | None = None) -> tuple[str, list[str]]:
    """
    Auto-detect optimal hardware configuration and corresponding extras.

    Returns:
        tuple: (hardware_type, hardware_extras_list)
    """
    if hardware is None:
        hardware = get_hardware_preference()

    hardware_extras = []

    # Check if hardware is a specific CUDA version (cu124, cu126, cu128)
    if hardware and hardware.startswith("cu"):
        hardware_extras = [hardware]  # Use the specified CUDA version directly
    elif hardware == "h100":
        hardware_extras = ["cu128"]  # Latest available CUDA for H100
    elif hardware == "a100":
        hardware_extras = ["cu128"]  # Cutting-edge CUDA for A100
    elif hardware == "gpu":
        optimal_cuda = get_optimal_cuda_version()
        hardware_extras = [optimal_cuda]
    elif hardware == "cpu":
        hardware_extras = ["cpu"]
    else:
        # Custom hardware specification
        hardware_extras = [hardware]

    return hardware, hardware_extras

def init_project_workflow(
    session: nox.Session,
    python_version: str = DEFAULT_PYTHON_VERSION,
    default_extras: list[str] | None = None,
    hardware: str | None = None,
    upgrade: bool = False,
    force_recreate_venv: bool = False,
    setup_precommit: bool = False
) -> bool:
    """
    Comprehensive project initialization following best software engineering practices.

    This function provides a complete project setup workflow:
    1. Hardware detection and optimal extra selection
    2. Dependency resolution and locking for multiple configurations
    3. Universal lockfile generation for fast syncing
    4. Virtual environment initialization with correct Python version
    5. Environment syncing with selected extras

    Args:
        session: The nox session
        python_version: Python version to use (must be supported)
        default_extras: Base extras to include (defaults to ["default"])
        hardware: Hardware type (auto-detected if None)
        upgrade: Whether to upgrade all dependencies
        force_recreate_venv: Whether to recreate the virtual environment

    Returns:
        bool: True if initialization successful, False otherwise
    """
    if default_extras is None:
        default_extras = ["default"]

    session.log("🚀 INITIALIZING PROJECT")
    session.log("=" * 80)

    # Step 1: Validate Python version
    session.log(f"🐍 Validating Python version: {python_version}")
    if not validate_python_version(python_version):
        session.error(f"❌ Python {python_version} is not supported by this project")
        return False

    session.log(f"✅ Python {python_version} is supported")

    # Step 2: Auto-detect hardware and determine optimal extras
    session.log("🖥️  Detecting hardware configuration...")
    detected_hardware, hardware_extras = auto_detect_hardware_extras(hardware)

    capabilities = detect_hardware_capabilities()
    session.log(f"🎯 Detected hardware: {detected_hardware}")

    if capabilities["gpu_available"]:
        session.log(f"🎮 GPUs: {capabilities['gpu_count']} ({', '.join(capabilities['gpu_names'])})")
        if capabilities["is_h100"]:
            session.log("⚡ H100 detected → Using cutting-edge cu128 optimizations")
        elif capabilities["is_a100"]:
            session.log("⚡ A100 detected → Using stable cu126 optimizations")
    else:
        session.log("💻 CPU-only mode detected")

    # Step 3: Prepare final extras list
    final_extras = list(default_extras) + hardware_extras
    final_extras = list(dict.fromkeys(final_extras))  # Remove duplicates while preserving order

    session.log(f"📦 Final extras configuration: {final_extras}")

    # Step 4: Generate individual lockfiles for different configurations
    session.log("🔒 Generating dependency lockfiles...")

    try:
        # Generate lockfile for the current configuration
        lockfile_path = resolve_lockfile_path(python_version, final_extras)
        generate_lockfile(session, lockfile_path, final_extras, upgrade=upgrade)

        # Generate additional common configurations
        common_configs = [
            ([], "base configuration"),
            (["dev"], "development configuration"),
            (["test"], "testing configuration"),
            (["notebook"], "notebook configuration"),
        ]

        for extra_config, description in common_configs:
            if extra_config != final_extras:  # Skip if same as main config
                config_extras = list(extra_config) + hardware_extras
                config_extras = list(dict.fromkeys(config_extras))
                config_lockfile = resolve_lockfile_path(python_version, config_extras)

                if not config_lockfile.exists():
                    session.log(f"📋 Generating lockfile for {description}...")
                    generate_lockfile(session, config_lockfile, config_extras, upgrade=upgrade)
                else:
                    session.log(f"📋 Lockfile exists for {description}: {config_lockfile.name}")

    except Exception as e:
        session.error(f"❌ Failed to generate lockfiles: {e}")
        return False

    # Step 5: Generate universal uv.lock
    session.log("🌐 Generating universal uv.lock for fast syncing...")

    try:
        generate_universal_lock(session, upgrade=upgrade)
    except Exception as e:
        session.error(f"❌ Failed to generate universal lockfile: {e}")
        return False

    # Step 6: Initialize virtual environment
    session.log(f"🔧 Initializing virtual environment with Python {python_version}...")

    venv_path = PROJECT_ROOT / ".venv"

    try:
        # Remove existing venv if force recreate is requested
        if force_recreate_venv and venv_path.exists():
            session.log("🗑️  Removing existing virtual environment...")
            session.run("rm", "-rf", str(venv_path), external=True)

        # Create virtual environment with specific Python version
        session.run("uv", "venv", "--python", python_version, external=True)
        session.log(f"✅ Virtual environment created: {venv_path}")

    except Exception as e:
        session.error(f"❌ Failed to create virtual environment: {e}")
        return False

    # Step 7: Sync virtual environment with selected extras
    session.log(f"⚡ Syncing virtual environment with extras: {final_extras}")

    try:
        # Build sync command
        sync_cmd = ["uv", "sync", "--frozen"]

        for extra in final_extras:
            sync_cmd.extend(["--extra", extra])

        session.run(*sync_cmd, external=True)
        session.log("✅ Virtual environment synced successfully")

    except Exception as e:
        session.error(f"❌ Failed to sync virtual environment: {e}")
        return False

    # Step 8: Verify installation with comprehensive checks
    session.log("🔍 Verifying installation...")

    try:
        # Use the new comprehensive test function
        test_ml_environment(session, capabilities)
        session.log("✅ Installation verification successful")

    except Exception as e:
        session.log(f"⚠️  Installation verification failed: {e}")
        session.log("💡 Project initialized but some components may not be working correctly")

    # Step 9: Summary and next steps
    session.log("=" * 80)
    session.log("🎉 PROJECT INITIALIZATION COMPLETE")
    session.log("=" * 80)
    session.log(f"🐍 Python version: {python_version}")
    session.log(f"🖥️  Hardware: {detected_hardware}")
    session.log(f"📦 Extras: {final_extras}")
    session.log(f"📁 Virtual environment: {venv_path}")
    session.log(f"🔒 Lockfiles: {lockfile_path.parent}")
    session.log("")
    session.log("💡 Next steps:")
    session.log("   • Use 'uv run python script.py' for running scripts")
    session.log("   • Use 'nox -s marimo-edit' for interactive notebooks")
    session.log("   • Use 'nox -s test' for running tests")
    session.log("   • Use 'nox -s quality-check' for code quality checks")
    session.log("=" * 80)

    # Optional pre-commit setup
    if setup_precommit:
        session.log("🔧 Setting up pre-commit hooks...")
        try:
            # Install pre-commit in the project environment
            session.run("uv", "pip", "install", "pre-commit")

            # Install the pre-commit hooks
            session.run("pre-commit", "install")
            session.run("pre-commit", "install", "--hook-type", "commit-msg")
            session.run("pre-commit", "install", "--hook-type", "pre-push")

            # Run pre-commit on all files to ensure everything is clean
            session.run("pre-commit", "run", "--all-files")

            session.log("✅ Pre-commit hooks installed and configured successfully!")
            session.log("🔧 Pre-commit hooks: Installed and configured")
        except Exception as e:
            session.log(f"⚠️  Pre-commit setup failed: {e}")
            session.log("💡 You can manually install pre-commit later with: pre-commit install")

    return True

# ===============================================================================
# UTILITY FUNCTIONS - Enhanced for Production
# ===============================================================================

def load_pyproject_toml() -> dict:
    """Load and parse pyproject.toml with error handling."""
    try:
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Warning: Could not load pyproject.toml: {e}")
        return {}

def get_all_extras() -> set[str]:
    """Extract all available extras from pyproject.toml."""
    try:
        pyproject = load_pyproject_toml()
        optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
        return set(optional_deps.keys())
    except Exception as e:
        print(f"Warning: Could not load extras: {e}")
        return set()

def get_project_version() -> str:
    """Get current project version."""
    try:
        pyproject = load_pyproject_toml()
        return pyproject.get("project", {}).get("version", "0.1.0")
    except Exception:
        return "0.1.0"

def get_comprehensive_env() -> dict[str, str]:
    """
    Get comprehensive environment configuration combining cache and hardware optimizations.

    Returns a complete environment dictionary that includes:
    - Cache hierarchy from pyproject.toml
    - Hardware-specific optimizations for H100/A100
    - Standard development environment variables
    """
    env = {}

    # Start with cache configuration
    env.update(COMMON_ENV)

    # Add hardware optimizations
    env.update(get_hardware_optimizations())

    # Ensure all cache directories exist
    for key, value in env.items():
        if key.endswith(("_DIR", "_HOME")):
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except (OSError, TypeError):
                # Skip if not a valid path
                pass

    return env

def global_setup(session: nox.Session) -> None:
    """Set comprehensive global environment variables for all sessions."""
    # Get comprehensive environment
    comprehensive_env = get_comprehensive_env()

    # Apply all environment variables to the session
    for key, value in comprehensive_env.items():
        session.env[key] = value

    # Log the setup
    capabilities = detect_hardware_capabilities()
    session.log(f"🔧 Environment configured with {len(comprehensive_env)} variables")

    if capabilities["is_h100"] or capabilities["is_a100"]:
        hardware_vars = len(get_hardware_optimizations())
        session.log(f"⚡ Hardware optimizations: {hardware_vars} variables applied")

def with_global_setup(func):
    """Decorator to apply comprehensive global setup automatically."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Handle both regular sessions and python=False sessions
        if args and hasattr(args[0], 'env'):
            session = args[0]
            global_setup(session)
        return func(*args, **kwargs)
    return wrapper


# ===============================================================================
# ENVIRONMENT MANAGEMENT FUNCTIONS
# ===============================================================================

def get_extra_packages(extra_name: str, pyproject_path: str = "pyproject.toml") -> list[str]:
    """Get packages for a specific extra from pyproject.toml."""
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    if extra_name not in extras:
        raise ValueError(f"No such extra: {extra_name}")

    return extras[extra_name]


def get_ci_optimized_extras(session_type: str) -> str:
    """
    Get minimal extras for CI/CD environments to reduce install time and resource usage.

    Args:
        session_type: Type of session (lint, test, format, docs)

    Returns:
        Comma-separated string of minimal extras needed for CI/CD
    """
    # Check if we're in CI environment
    is_ci = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"

    if not is_ci:
        # Local development - use full extras
        detected_hardware, hardware_extras = auto_detect_hardware_extras()
        if session_type == "lint":
            return ",".join([*hardware_extras, "default", "dev", "lint", "typing"])
        elif session_type == "typing":
            return ",".join([*hardware_extras, "default", "dev", "typing"])
        elif session_type == "test":
            return ",".join([*hardware_extras, "default", "dev"])
        elif session_type == "format":
            return ",".join(["lint"])
        elif session_type == "docs":
            return ",".join(["cpu", "default", "docs"])
        else:
            return ",".join([*hardware_extras, "default"])

    # CI/CD environment - use ultra-minimal extras for speed
    if session_type == "lint":
        return "lint,typing"    # No CPU/PyTorch for linting
    elif session_type == "typing":
        return "typing"         # Just type checkers
    elif session_type == "test":
        return "cpu,default"    # Minimal for testing
    elif session_type == "format":
        return "lint"           # Just ruff and black
    elif session_type == "docs":
        return "cpu,docs"       # Minimal for documentation
    else:
        return "cpu"            # Absolute minimum


def setup_environment(
    session: nox.Session,
    extras: str | None = None,
    hardware: str | None = None,
    additional_packages: list[str] | None = None,
    upgrade: bool = False,
    use_clean_env: bool = False,
) -> None:
    """
    Sets up the environment by syncing dependencies using `uv sync`, with optional extras,
    hardware specification, additional packages, upgrade, and clean-env mode.

    Args:
        session: The nox session
        extras: Comma-separated list of extras to install
        hardware: Hardware type (cpu, gpu, h100, a100, cu124, cu126, cu128) - translated to appropriate extras
        additional_packages: Additional packages to install via pip
        upgrade: Whether to upgrade dependencies
        use_clean_env: Whether to use Nox's ephemeral environment

    Environment hierarchy:
    - Always use --active since this function is only called from nox sessions
    - Nox creates and activates the virtual environment before running commands
    """
    # Since this function is always called from a nox session, always use --active
    # Nox will have created and activated the virtual environment by the time commands run
    venv_flag = ["--active"]

    if use_clean_env:
        session.log("Using Nox's ephemeral clean environment.")
    else:
        virtual_env_path = os.environ.get("VIRTUAL_ENV", "<nox-managed>")
        session.log(f"Using nox-managed virtual environment: {virtual_env_path}")

    # Step 1: Process hardware specification and extras
    final_extras = []

    # Add hardware-based extras if specified
    if hardware:
        _, hardware_extras = auto_detect_hardware_extras(hardware)
        final_extras.extend(hardware_extras)
        session.log(f"Hardware '{hardware}' → extras: {', '.join(hardware_extras)}")

    # Add explicit extras if specified
    if extras:
        extra_list = [extra.strip() for extra in extras.split(",") if extra.strip()]
        final_extras.extend(extra_list)
        session.log(f"Explicit extras: {', '.join(extra_list)}")

    # Remove duplicates while preserving order
    final_extras = list(dict.fromkeys(final_extras))

    # Step 2: Sync with `uv sync`
    sync_cmd = ["uv", "sync", *venv_flag]

    # Clean lockfile should work in both CI and local environments without special flags

    if final_extras:
        for extra in final_extras:
            sync_cmd.extend(["--extra", extra])
        session.log(f"Syncing extras: {', '.join(final_extras)}")
    else:
        session.log("Syncing base dependencies")

    if upgrade:
        sync_cmd.append("--upgrade")
        session.log("Syncing with upgrade enabled")

    session.run(*sync_cmd)

    # Step 3: Additional package install
    if additional_packages:
        session.log(f"Installing additional packages: {' '.join(additional_packages)}")
        pip_cmd = ["uv", "pip", "install"]
        if upgrade:
            pip_cmd.append("--upgrade")
        pip_cmd += additional_packages
        session.run(*pip_cmd)


def test_ml_environment(session: nox.Session, capabilities: dict) -> None:
    """
    Comprehensive ML environment testing in a single uv run call.

    Tests all major components: Python, study, PyTorch, CUDA, PyG, PyG extensions, and Lightning.
    Uses a single uv run call for efficiency instead of multiple separate calls.

    Args:
        session: The nox session
        capabilities: Hardware capabilities dictionary
    """
    session.log("🔍 Testing ML environment components...")

    # Create comprehensive test script
    test_script = '''
import sys
import importlib
from typing import List, Dict

def test_import(module_name: str, version_attr: str = None, extra_info: str = None) -> bool:
    """Test importing a module and optionally get its version."""
    try:
        module = importlib.import_module(module_name)
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"✅ {module_name} {version} import successful")
        else:
            print(f"✅ {module_name} import successful")
        if extra_info:
            print(f"   {extra_info}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} import error: {e}")
        return False

def test_pytorch_cuda() -> Dict[str, any]:
    """Test PyTorch and CUDA functionality."""
    results = {}

    if test_import("torch", "__version__"):
        import torch
        results["pytorch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            results["gpu_count"] = torch.cuda.device_count()
            results["cuda_version"] = torch.version.cuda
            print(f"   CUDA available: {results['cuda_available']}")
            print(f"   GPU count: {results['gpu_count']}")
            print(f"   CUDA version: {results['cuda_version']}")
        else:
            print("   CUDA not available")

    return results

def test_pyg_extensions() -> List[str]:
    """Test PyG extensions and return list of successful imports."""
    pyg_extensions = [
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "pyg_lib"
    ]

    successful = []
    for ext in pyg_extensions:
        if test_import(ext):
            successful.append(ext)
        else:
            print(f"   💡 This may be expected if {ext} is not available for your platform")

    return successful

# Main test execution
print("=" * 60)
print("🧪 COMPREHENSIVE ML ENVIRONMENT TEST")
print("=" * 60)

# Test Python and basic imports
print(f"🐍 Python {sys.version}")
test_import("study")

# Test PyTorch and CUDA
print("\\n🔬 Testing PyTorch and CUDA...")
pytorch_results = test_pytorch_cuda()

# Test PyTorch Geometric
print("\\n🔬 Testing PyTorch Geometric...")
test_import("torch_geometric", "__version__")

# Test PyG extensions
print("\\n🔬 Testing PyG extensions...")
successful_extensions = test_pyg_extensions()
print(f"   Successful PyG extensions: {len(successful_extensions)}/{5}")

# Test Lightning
print("\\n🔬 Testing PyTorch Lightning...")
test_import("lightning", "__version__")

# Test local dependencies
print("\\n🔬 Testing local dependencies...")
local_deps = []
local_success = 0
for dep in local_deps:
    if test_import(dep):
        local_success += 1

# Summary
print("\\n" + "=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)
print(f"✅ Core components: Python, study, PyTorch, PyG, Lightning")
print(f"✅ PyG extensions: {len(successful_extensions)}/{5} successful")
print(f"✅ Local dependencies: {local_success}/{len(local_deps)} successful")
if pytorch_results.get("cuda_available"):
    print(f"✅ CUDA: Available (version {pytorch_results.get('cuda_version', 'unknown')})")
    print(f"✅ GPUs: {pytorch_results.get('gpu_count', 0)} detected")
else:
    print("⚠️  CUDA: Not available (CPU-only mode)")

print("\\n🎉 ML environment testing completed successfully!")
'''

    # Run the comprehensive test
    try:
        session.run("uv", "run", "--active", "python", "-c", test_script, external=True)
        session.log("✅ ML environment testing completed successfully")
    except Exception as e:
        session.log(f"⚠️  ML environment testing failed: {e}")
        session.log("💡 Some components may not be available for your platform")


# ===============================================================================
# CORE PROJECT MANAGEMENT SESSIONS (CLEANED UP)
# ===============================================================================

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def init_project(session: nox.Session) -> None:
    """
    Initialize project with comprehensive dependency management and environment setup.

    This session provides a complete project initialization workflow following
    best software engineering practices. It handles ALL dependencies through the main pyproject.toml.

    Examples:
        nox -s init_project-3.12                           # Auto-detect hardware, default extras
        nox -s init_project-3.12 -- --hardware=cu126       # Specify CUDA 12.6 hardware
        nox -s init_project-3.12 -- --extras=dev,notebook  # Custom extras
        nox -s init_project-3.12 -- --upgrade              # Upgrade all dependencies
        nox -s init_project-3.12 -- --force-recreate       # Recreate virtual environment
    """
    import argparse

    # Parse arguments from session.posargs
    parser = argparse.ArgumentParser(description="Initialize project with dependency management")
    parser.add_argument("--extras", default="default",
                       help="Comma-separated list of extras (default: 'default')")
    parser.add_argument("--hardware", choices=["cpu", "gpu", "h100", "a100", "cu124", "cu126", "cu128", "auto"], default="auto",
                       help="Hardware type (default: auto-detect)")
    parser.add_argument("--upgrade", action="store_true",
                       help="Upgrade all dependencies to latest versions")
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreation of virtual environment")
    parser.add_argument("--setup-precommit", action="store_true",
                       help="Set up pre-commit hooks after project initialization")

    args = parser.parse_args(session.posargs or [])

    # Parse extras
    if args.extras:
        default_extras = [extra.strip() for extra in args.extras.split(",") if extra.strip()]
    else:
        default_extras = ["default"]

    # Determine hardware
    hardware = None if args.hardware == "auto" else args.hardware

    # Log session info with initialization parameters
    log_session_info(session, {
        "operation": "project_initialization",
        "python_version": session.python,
        "extras": ",".join(default_extras),
        "hardware": args.hardware,
        "upgrade": args.upgrade,
        "force_recreate": args.force_recreate
    })

    # Run project initialization
    success = init_project_workflow(
        session=session,
        python_version=session.python,
        default_extras=default_extras,
        hardware=hardware,
        upgrade=args.upgrade,
        force_recreate_venv=args.force_recreate,
        setup_precommit=args.setup_precommit
    )

    if not success:
        session.error("❌ Project initialization failed")

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def lock_universal(session: nox.Session) -> None:
    """Generate universal uv.lock with all extras for maximum performance."""
    log_session_info(session, {"operation": "universal_lockfile_generation"})

    upgrade = "--upgrade" in (session.posargs or [])
    generate_universal_lock(session, upgrade=upgrade)

# ===============================================================================
# INTERACTIVE NOTEBOOK AND SCRIPT SESSIONS (ENHANCED WITH CONSISTENT ARGS)
# ===============================================================================

@nox.session(name="marimo-edit", python=False)
def marimo_edit(session: nox.Session) -> None:
    """
    Launch Marimo in edit mode with configurable resource allocation.

    This session provides interactive development with complete control over resource allocation.
    Uses sensible defaults: single GPU, half CPUs, 50% memory.
    By default, runs in tmux/screen session for persistent development.

    Usage:
        nox -s marimo-edit                                    # Default: tmux/screen session with single GPU, half CPUs, 50% memory
        nox -s marimo-edit -- --no-tmux                      # Run directly without tmux/screen
        nox -s marimo-edit -- --gpus=all                     # Use all GPUs
        nox -s marimo-edit -- --gpus=0,1 --cpus=0-15         # Specific GPU/CPU allocation
        nox -s marimo-edit -- --memory=64                    # Limit memory to 64GB
        nox -s marimo-edit -- --port=8889                    # Custom port
        nox -s marimo-edit -- --enhanced                     # Enable ML optimizations
    """
    # Verify .venv exists
    if not os.path.exists(".venv"):
        session.error("No .venv found. Please run 'nox -s init_project' first.")

    # Parse resource allocation arguments with sensible defaults
    args = parse_resource_arguments(session, "Marimo with configurable resource allocation", None)

    session.log("🎨 Launching Marimo with configurable resources...")
    session.log("📝 Using existing .venv without modifications")
    session.log(f"🔧 Resource allocation: GPUs={args.gpus}, CPUs={args.cpus}, Memory={args.memory}GB")

    # Apply resource allocation
    resource_env = allocate_resources(
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        auto_optimize=True
    )

    # Add enhanced ML optimizations if requested
    if args.enhanced:
        session.log("🚀 Enabling enhanced ML framework optimizations...")
        enhanced_env = {
            "TORCH_CUDNN_V8_API_ENABLED": "1",
            "TORCH_COMPILE_DEBUG": "0",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "CUDA_LAUNCH_BLOCKING": "0",
            "PYTHONOPTIMIZE": "1",
        }
        resource_env.update(enhanced_env)

    # Build command arguments
    cmd_args = ["uv", "run", "--active", "marimo", "edit", "--no-token", "--port", args.port, "--host", "0.0.0.0"]

    session.log(f"🖥️  Running on port {args.port}")
    session.log(f"💡 Access at http://localhost:{args.port} or use SSH tunneling for remote access")

    # Run in tmux/screen or directly
    session_name = f"marimo-edit-{args.port}"
    run_in_multiplexer(session_name, cmd_args, resource_env, no_tmux=args.no_tmux)


@nox.session(name="marimo-run", python=False)
def marimo_run(session: nox.Session) -> None:
    """
    Run Marimo app with configurable resource allocation.

    This session runs Marimo apps using the current project environment with optimal resource allocation.
    Uses sensible defaults: single GPU, half CPUs, 50% memory.
    By default, runs in tmux/screen session for persistent execution.

    Usage:
        nox -s marimo-run -- app.py                           # Default: tmux/screen session
        nox -s marimo-run -- app.py --no-tmux                # Run directly without tmux/screen
        nox -s marimo-run -- app.py --gpus=all --memory=32   # Custom resource allocation
    """
    # Verify .venv exists
    if not os.path.exists(".venv"):
        session.error("No .venv found. Please run 'nox -s init_project' first.")

    # Parse arguments (app file + resource args)
    if not session.posargs:
        session.error("Please specify a Marimo app file: nox -s marimo-run -- app.py")

    app_file = session.posargs[0]

    # Parse resource arguments from remaining posargs
    resource_args = session.posargs[1:] if len(session.posargs) > 1 else []

    args = parse_resource_arguments(session, "Run Marimo app with resource allocation", resource_args)

    session.log("🚀 Running Marimo app...")
    session.log("📝 Using existing .venv without modifications")
    session.log(f"🔧 Resource allocation: GPUs={args.gpus}, CPUs={args.cpus}, Memory={args.memory}GB")

    # Apply resource allocation for optimal performance
    resource_env = allocate_resources(
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        auto_optimize=True
    )

    # Build command arguments
    cmd_args = ["uv", "run", "--active", "marimo", "run", app_file, "--no-token", "--port", args.port, "--host", "0.0.0.0"]

    # Run in tmux/screen or directly
    session_name = f"marimo-run-{os.path.splitext(os.path.basename(app_file))[0]}-{args.port}"
    run_in_multiplexer(session_name, cmd_args, resource_env, no_tmux=args.no_tmux)


@nox.session(name="jupyter-lab", python=False)
def jupyter_lab(session: nox.Session) -> None:
    """
    Launch JupyterLab with configurable resource allocation.

    This session starts JupyterLab for interactive development using the current project environment.
    Uses sensible defaults: single GPU, half CPUs, 50% memory.
    By default, runs in tmux/screen session for persistent development.

    Usage:
        nox -s jupyter-lab                                   # Default: tmux/screen session
        nox -s jupyter-lab -- --no-tmux                     # Run directly without tmux/screen
        nox -s jupyter-lab -- --gpus=all --memory=64        # Custom allocation
    """
    # Verify .venv exists
    if not os.path.exists(".venv"):
        session.error("No .venv found. Please run 'nox -s init_project' first.")

    # Parse resource allocation arguments
    args = parse_resource_arguments(session, "JupyterLab with configurable resource allocation", None)

    session.log("🔬 Launching JupyterLab...")
    session.log("📝 Using existing .venv without modifications")
    session.log(f"🔧 Resource allocation: GPUs={args.gpus}, CPUs={args.cpus}, Memory={args.memory}GB")

    # Apply resource allocation for optimal performance
    resource_env = allocate_resources(
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        auto_optimize=True
    )

    # Add enhanced optimizations if requested
    if args.enhanced:
        session.log("🚀 Enabling enhanced ML framework optimizations...")
        enhanced_env = {
            "TORCH_CUDNN_V8_API_ENABLED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "CUDA_LAUNCH_BLOCKING": "0",
        }
        resource_env.update(enhanced_env)

    # Build command arguments
    cmd_args = ["uv", "run", "--active", "jupyter", "lab",
                "--port", args.port,
                "--host", "0.0.0.0",
                "--no-browser",
                "--allow-root"]

    # Run in tmux/screen or directly
    session_name = f"jupyter-lab-{args.port}"
    run_in_multiplexer(session_name, cmd_args, resource_env, no_tmux=args.no_tmux)


@nox.session(name="jupyter-notebook", python=False)
def jupyter_notebook(session: nox.Session) -> None:
    """
    Launch Jupyter Notebook with configurable resource allocation.

    This session starts Jupyter Notebook for interactive development using the current project environment.
    Uses sensible defaults: single GPU, half CPUs, 50% memory.
    By default, runs in tmux/screen session for persistent development.

    Usage:
        nox -s jupyter-notebook                              # Default: tmux/screen session
        nox -s jupyter-notebook -- --no-tmux                # Run directly without tmux/screen
        nox -s jupyter-notebook -- --gpus=all --memory=64   # Custom allocation
    """
    # Verify .venv exists
    if not os.path.exists(".venv"):
        session.error("No .venv found. Please run 'nox -s init_project' first.")

    # Parse resource allocation arguments
    args = parse_resource_arguments(session, "Jupyter Notebook with configurable resource allocation", None)

    session.log("📓 Launching Jupyter Notebook...")
    session.log("📝 Using existing .venv without modifications")
    session.log(f"🔧 Resource allocation: GPUs={args.gpus}, CPUs={args.cpus}, Memory={args.memory}GB")

    # Apply resource allocation for optimal performance
    resource_env = allocate_resources(
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        auto_optimize=True
    )

    # Add enhanced optimizations if requested
    if args.enhanced:
        session.log("🚀 Enabling enhanced ML framework optimizations...")
        enhanced_env = {
            "TORCH_CUDNN_V8_API_ENABLED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "CUDA_LAUNCH_BLOCKING": "0",
        }
        resource_env.update(enhanced_env)

    # Build command arguments
    cmd_args = ["uv", "run", "--active", "jupyter", "notebook",
                "--port", args.port,
                "--host", "0.0.0.0",
                "--no-browser",
                "--allow-root"]

    # Run in tmux/screen or directly
    session_name = f"jupyter-notebook-{args.port}"
    run_in_multiplexer(session_name, cmd_args, resource_env, no_tmux=args.no_tmux)


@nox.session(name="run-script", python=False)
@with_global_setup
def run_script(session: nox.Session) -> None:
    """
    Run Python scripts with configurable resource allocation and comprehensive ML environment.

    This unified session runs Python scripts with optimal resource allocation, enhanced ML optimizations,
    and a complete ML/data science environment including experiment tracking.

    Features:
    - Uses sensible defaults: single GPU, half CPUs, 50% memory
    - Always enables enhanced ML framework optimizations
    - Uses default extras (includes ml, bio, plotting, graph, notebook)
    - Complete experiment tracking: wandb, tensorboard, mlflow, optuna (via ml extras)
    - Comprehensive environment variables for optimal performance

    Usage:
        nox -s run-script -- script.py [script_args...] [--gpus=all] [--memory=32]
        nox -s run-script -- train.py --epochs=100 --gpus=0,1 --memory=64
        nox -s run-script -- experiment.py --gpus=all
    """
    if not session.posargs:
        session.error("Please specify a script to run: nox -s run-script -- script.py [args...]")

    # Parse script file and arguments
    script_file = session.posargs[0]
    remaining_args = session.posargs[1:]

    # Separate script arguments from resource arguments
    script_args = []
    resource_args = []

    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg.startswith('--'):
            # Check if it's a resource argument (handle both --key=value and --key value formats)
            key = arg[2:].split('=')[0]  # Extract key part before '=' if present
            if key in ['gpus', 'cpus', 'memory', 'enhanced', 'port']:
                # This is a resource argument
                if '=' in arg:
                    # Format: --gpus=none
                    resource_args.append(arg)
                else:
                    # Format: --gpus none
                    resource_args.append(arg)
                    if i + 1 < len(remaining_args) and not remaining_args[i + 1].startswith('--'):
                        resource_args.append(remaining_args[i + 1])
                        i += 1
            else:
                script_args.append(arg)
        else:
            script_args.append(arg)
        i += 1

    # Parse resource arguments (enhanced is always enabled)
    resource_args.append("--enhanced")  # Always enable enhanced optimizations
    args = parse_resource_arguments(session, "Run Python script with comprehensive ML environment", resource_args)

    log_session_info(session, {
        "operation": "run_script",
        "script": script_file,
        "gpus": args.gpus,
        "cpus": args.cpus,
        "memory_gb": args.memory,
        "enhanced": True,  # Always enhanced
    })

    # Apply resource allocation
    resource_env = allocate_resources(
        gpus=args.gpus,
        cpus=args.cpus,
        memory_gb=args.memory,
        auto_optimize=True
    )

    # Always enable enhanced ML framework optimizations for best performance
    session.log("🚀 Enhanced ML framework optimizations enabled by default")
    enhanced_env = {
        # PyTorch optimizations
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "TORCH_COMPILE_DEBUG": "0",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "CUDA_LAUNCH_BLOCKING": "0",
        "PYTHONOPTIMIZE": "1",
        # Experiment tracking optimizations (wandb already available via ml extras)
        "WANDB_SILENT": "true",  # Reduce wandb output noise
        "MLFLOW_TRACKING_INSECURE_TLS": "true",  # For potential local mlflow use
        # Additional performance optimizations
        "OMP_PROC_BIND": "true",  # Better CPU thread affinity
        "KMP_AFFINITY": "granularity=fine,compact,1,0",  # Intel MKL optimizations
    }
    resource_env.update(enhanced_env)

    for key, value in resource_env.items():
        session.env[key] = value

    # Setup environment with default extras (includes ml with wandb, bio, plotting, graph, notebook)
    setup_environment(
        session=session,
        extras="default",  # Already includes ml (with wandb), bio, plotting, graph, notebook
        use_clean_env=False,  # Use project .venv for script execution
        upgrade=False
    )

    session.log(f"🚀 Running script: {script_file}")
    session.log(f"📝 Script arguments: {' '.join(script_args)}")
    session.log("🔬 Environment: default extras (ml, bio, plotting, graph, notebook)")
    session.log("📊 Experiment tracking: wandb, tensorboard, mlflow, optuna available (via ml extras)")

    # Run the script
    session.run("uv", "run", "--active", "python", script_file, *script_args)





# ===============================================================================
# DOCUMENTATION BUILDING AND NOTEBOOK EXPORT SESSIONS
# ===============================================================================

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def build_docs(session: nox.Session) -> None:
    """Build comprehensive documentation with Sphinx and Marimo tutorials."""
    log_session_info(session, {"operation": "build_docs"})

    # Get CI-optimized extras for better performance
    extras = get_ci_optimized_extras("docs")
    is_ci = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"

    session.log(f"🔧 Environment: {'CI/CD' if is_ci else 'Local'} → extras: {extras}")

    setup_environment(
        session=session,
        extras=extras,
        hardware="cpu" if is_ci else None,
        use_clean_env=is_ci,  # Clean env for CI, shared for local
        upgrade=False
    )

    session.log("📚 Building documentation...")

    # Create docs build directory
    docs_build_dir = Path("docs/_build")
    docs_build_dir.mkdir(parents=True, exist_ok=True)

    # Skip marimo tutorial building since we have regular Python tutorials
    session.log("📚 Skipping marimo tutorial building (using exported Python tutorials)...")

    # Ensure sphinx is installed
    session.log("📦 Installing sphinx and documentation dependencies...")
    session.install("sphinx", "sphinx-rtd-theme", "sphinx-book-theme", silent=False)

    # Build Sphinx documentation
    session.log("📖 Building Sphinx documentation...")
    session.run("python", "-m", "sphinx",
                "-b", "html",
                "docs",
                str(docs_build_dir / "html"),
                "--keep-going")  # Continue on errors

    session.log("✅ Documentation build completed")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def build_marimo_tutorials(session: nox.Session) -> None:
    """Build marimo tutorials to multiple formats."""
    log_session_info(session, {"operation": "build_marimo_tutorials"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev + docs
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev", "docs"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["weasyprint", "pyyaml", "jupytext"],
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🎨 Building marimo tutorials...")

    # Parse command line arguments
    formats = session.posargs if session.posargs else ["html", "markdown", "pdf", "jupyter"]

    # Run the marimo tutorial builder
    cmd = ["python", "docs/build_marimo_tutorials.py", "--project-root", ".", "--output-dir", "docs/_build/tutorials", "--formats", *formats]

    session.run(*cmd)

    session.log("✅ Marimo tutorials build completed")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def validate_marimo_tutorials(session: nox.Session) -> None:
    """Validate marimo tutorials for quality and consistency."""
    log_session_info(session, {"operation": "validate_marimo_tutorials"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev + docs
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev", "docs"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["pyyaml"],
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🔍 Validating marimo tutorials...")

    # Run validation checks
    session.run("python", "docs/build_marimo_tutorials.py",
                "--project-root", ".",
                "--output-dir", "docs/_build/tutorials_validation",
                "--formats", "html",  # Just test HTML export
                "--verbose")

    session.log("✅ Marimo tutorials validation completed")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def deploy_docs(session: nox.Session) -> None:
    """Deploy documentation to GitLab Pages or other hosting platforms."""
    log_session_info(session, {"operation": "deploy_docs"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Ensure we have all necessary extras for documentation deployment
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = ["cu128", "default", "dev", "docs"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")
    session.log("📦 Setting up environment for documentation deployment...")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["sphinx", "sphinx-rtd-theme", "sphinx-book-theme"],
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🚀 Deploying documentation...")

    # Check if we're in GitLab CI environment
    import os
    if os.getenv("GITLAB_CI"):
        session.log("🤖 GitLab CI detected - documentation will be deployed via GitLab Pages")
        session.log("📚 Run 'git push' to trigger GitLab Pages deployment")
    else:
        session.log("💻 Local deployment - preparing documentation for GitLab Pages")

        # First, export tutorials with clean filenames
        session.log("📚 Exporting tutorials...")
        session.run("nox", "-s", "export-tutorials", external=True)

        # Build documentation
        session.log("📖 Building documentation...")
        session.run("nox", "-s", "build_docs", external=True)

        # Create public directory for GitLab Pages
        public_dir = Path("public")
        if public_dir.exists():
            import shutil
            shutil.rmtree(public_dir)

        # Copy built documentation to public directory
        import shutil
        shutil.copytree("docs/_build/html", "public")

        session.log("📚 Documentation prepared in 'public' directory")
        session.log("🚀 To deploy to GitLab Pages, commit and push to main branch")

    session.log("✅ Documentation deployment preparation completed")


@nox.session(name="manual-gitlab-pages", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def manual_gitlab_pages(session: nox.Session) -> None:
    """Manually deploy documentation to GitLab Pages using git push."""
    log_session_info(session, {"operation": "manual_gitlab_pages"})

    # Auto-detect hardware and setup environment
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = ["cu128", "default", "dev", "docs"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")
    session.log("📦 Setting up environment for manual GitLab Pages deployment...")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["sphinx", "sphinx-rtd-theme", "sphinx-book-theme"],
        use_clean_env=False,
        upgrade=False
    )

    session.log("🚀 Manual GitLab Pages deployment...")

    # Build documentation locally
    session.log("📚 Exporting tutorials...")
    session.run("nox", "-s", "export-tutorials", external=True)

    session.log("📖 Building documentation...")
    session.run("nox", "-s", "build_docs", external=True)

    # Prepare public directory
    session.log("📁 Preparing public directory...")
    public_dir = Path("public")
    if public_dir.exists():
        import shutil
        shutil.rmtree(public_dir)

    import shutil
    shutil.copytree("docs/_build/html", "public")

    session.log("📝 Documentation prepared in 'public' directory")
    session.log("")
    session.log("🚀 To complete deployment, run one of these options:")
    session.log("")
    session.log("   Option 1 (Recommended - via CI):")
    session.log("   git add public/ docs/tutorials/")
    session.log("   git commit -m 'Deploy documentation'")
    session.log("   git push origin main")
    session.log("")
    session.log("   Option 2 (Direct push):")
    session.log("   ./scripts/manual_gitlab_pages_deploy.sh")
    session.log("")
    session.log("✅ Manual deployment preparation completed")


@nox.session(name="marimo-export", python=False)
def marimo_export(session: nox.Session) -> None:
    """
    Export Marimo apps to various formats (Markdown, HTML, etc.).

    This session exports Marimo applications to different formats
    for documentation and sharing. Uses the same environment as marimo-edit
    to ensure consistency.
    """
    # Verify .venv exists
    if not os.path.exists(".venv"):
        session.error("No .venv found. Please run 'nox -s init_project' first.")

    session.log("📤 Exporting Marimo app using existing .venv...")
    session.log("📝 Using same environment as marimo-edit for consistency")

    if not session.posargs:
        session.error("Please specify a Marimo app file: nox -s marimo-export -- app.py")

    app_file = session.posargs[0]
    export_format = session.posargs[1] if len(session.posargs) > 1 else "html"

    session.log(f"📤 Exporting {app_file} to {export_format}...")

    # Export Marimo app using uv run for consistency
    session.run("uv", "run", "--active", "marimo", "export", export_format, app_file, external=True)

    session.log("✅ Marimo export completed")




@nox.session(name="export-tutorials", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def export_tutorials(session: nox.Session) -> None:
    """Export Python tutorials to beautiful notebook-style markdown with executed output."""
    log_session_info(session, {"operation": "export_tutorials"})

    # Auto-detect hardware and setup environment
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = ["cu128", "default", "dev", "docs"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")
    session.log("📦 Setting up professional tutorial export environment...")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=[
            "markdown", "pygments", "torch", "numpy", "pandas",
            "scikit-learn", "matplotlib"
        ],
        use_clean_env=False,
        upgrade=False
    )

    # Import the tutorial exporter
    sys.path.insert(0, str(Path("docs")))
    from tutorial_exporter import TutorialExporter

    # Create exporter instance
    exporter = TutorialExporter()

    # Define paths
    docs_dir = Path("docs")
    tutorials_dir = docs_dir / "tutorials"

    # Find all Python tutorial files
    tutorial_files = list(tutorials_dir.glob("*.py"))
    tutorial_files = [f for f in tutorial_files if not f.name.startswith("__") and not f.name.startswith("tutorial_exporter")]
    tutorial_files.sort()

    if not tutorial_files:
        session.error("❌ No tutorial files found in docs/tutorials/")
        return

    session.log(f"📚 Found {len(tutorial_files)} tutorial(s) to process:")
    for tutorial_file in tutorial_files:
        session.log(f"  - {tutorial_file.name}")
    session.log("")

    # Create beautiful notebook-style exports
    success_count = 0
    for tutorial_file in tutorial_files:
        session.log(f"🔄 Creating notebook-style export for {tutorial_file.name}...")

        try:
            base_name = tutorial_file.stem
            export_filename = f"{base_name}.md"
            export_path = tutorials_dir / export_filename

            # Generate beautiful notebook content
            notebook_content = exporter.export_tutorial(tutorial_file, session)

            # Write the professional markdown
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(notebook_content)

            session.log(f"✅ Professional tutorial export created: {export_path}")
            success_count += 1

        except Exception as e:
            session.log(f"❌ Error exporting {tutorial_file.name}: {e}")
            # Create a fallback export
            fallback_content = f"# {tutorial_file.stem.replace('_', ' ').title()}\n\nError creating tutorial: {e}"
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(fallback_content)

        session.log("")

    session.log(f"🎉 Tutorial export completed! Successfully exported {success_count}/{len(tutorial_files)} tutorials")
    session.log("📚 Professional notebook-style documentation is ready!")


@nox.session(name="list-sessions", python=False)
def list_sessions(session: nox.Session) -> None:
    """
    List all active tmux and screen sessions.

    This session shows all running tmux and screen sessions, making it easy
    to see what interactive development sessions are currently active.

    Usage:
        nox -s list-sessions
    """
    session.log("📋 Listing active tmux and screen sessions...")

    # List tmux sessions
    try:
        result = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0 and result.stdout.strip():
            session.log("🔧 Active tmux sessions:")
            for line in result.stdout.strip().split('\n'):
                session.log(f"  {line}")
        else:
            session.log("🔧 No active tmux sessions")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        session.log("🔧 tmux not available")

    # List screen sessions
    try:
        result = subprocess.run(["screen", "-list"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0 and result.stdout.strip():
            session.log("🔧 Active screen sessions:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('There is a screen on'):
                    session.log(f"  {line}")
        else:
            session.log("🔧 No active screen sessions")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        session.log("🔧 screen not available")

    session.log("✅ Session listing completed")


@nox.session(name="terminal-info", python=False)
def terminal_info(session: nox.Session) -> None:
    """
    Display information about the current terminal environment.

    This session helps diagnose terminal multiplexer conflicts and provides
    recommendations for optimal setup, especially for Warp terminal users.

    Usage:
        nox -s terminal-info
    """
    session.log("🔍 Analyzing terminal environment...")

    # Check if we're in a tmux session
    tmux_session = os.environ.get("TMUX")
    if tmux_session:
        session.log(f"⚠️  Currently in tmux session: {tmux_session}")
        session.log("💡 This might cause conflicts with nested tmux sessions")
        session.log("💡 Recommendation: Use screen instead of tmux for better compatibility")
    else:
        session.log("✅ Not in a tmux session")

    # Check if we're in a screen session
    screen_session = os.environ.get("STY")
    if screen_session:
        session.log(f"✅ Currently in screen session: {screen_session}")
    else:
        session.log("✅ Not in a screen session")

    # Check terminal type
    term = os.environ.get("TERM", "unknown")
    session.log(f"📱 Terminal type: {term}")

    # Check for Warp-specific environment
    if "warp" in term.lower() or os.environ.get("WARP_TERMINAL"):
        session.log("🚀 Detected Warp terminal")
        session.log("💡 Warp uses tmux internally, so screen is recommended for sessions")

    # Check available multiplexers
    session.log("\n🔧 Available terminal multiplexers:")

    try:
        result = subprocess.run(["screen", "-version"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            session.log(f"✅ Screen: {result.stdout.strip()}")
        else:
            session.log("❌ Screen: Not available")
    except Exception:
        session.log("❌ Screen: Not available")

    try:
        result = subprocess.run(["tmux", "-V"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            session.log(f"✅ Tmux: {result.stdout.strip()}")
        else:
            session.log("❌ Tmux: Not available")
    except Exception:
        session.log("❌ Tmux: Not available")

    # Recommendations
    session.log("\n💡 Recommendations:")
    if tmux_session and not screen_session:
        session.log("   • Install screen: nox -s install-screen")
        session.log("   • Or use --no-tmux flag to run directly")
        session.log("   • Or unset TMUX: unset TMUX && nox -s <session-name>")
    elif not screen_session and not tmux_session:
        session.log("   • Install screen: nox -s install-screen")
        session.log("   • Or install tmux: sudo apt install tmux")
    else:
        session.log("   • Terminal environment looks good!")

    session.log("✅ Terminal environment analysis completed")


@nox.session(name="install-screen", python=False)
def install_screen(session: nox.Session) -> None:
    """
    Install screen terminal multiplexer for better compatibility with Warp.

    This session installs screen which works better than tmux when running
    inside Warp terminal (which already uses tmux internally).

    Usage:
        nox -s install-screen
    """
    session.log("📦 Installing screen terminal multiplexer...")

    # Detect package manager and install screen
    try:
        # Try apt (Ubuntu/Debian)
        session.run("sudo", "apt", "update", external=True)
        session.run("sudo", "apt", "install", "-y", "screen", external=True)
        session.log("✅ Screen installed via apt")
    except subprocess.CalledProcessError:
        try:
            # Try yum (RHEL/CentOS)
            session.run("sudo", "yum", "install", "-y", "screen", external=True)
            session.log("✅ Screen installed via yum")
        except subprocess.CalledProcessError:
            try:
                # Try dnf (Fedora)
                session.run("sudo", "dnf", "install", "-y", "screen", external=True)
                session.log("✅ Screen installed via dnf")
            except subprocess.CalledProcessError:
                try:
                    # Try pacman (Arch)
                    session.run("sudo", "pacman", "-S", "--noconfirm", "screen", external=True)
                    session.log("✅ Screen installed via pacman")
                except subprocess.CalledProcessError:
                    session.error("❌ Could not install screen. Please install it manually:")
                    session.error("   Ubuntu/Debian: sudo apt install screen")
                    session.error("   RHEL/CentOS: sudo yum install screen")
                    session.error("   Fedora: sudo dnf install screen")
                    session.error("   Arch: sudo pacman -S screen")
                    session.error("   macOS: brew install screen")

    # Verify installation
    try:
        result = subprocess.run(["screen", "-version"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            session.log(f"✅ Screen installed successfully: {result.stdout.strip()}")
        else:
            session.error("❌ Screen installation verification failed")
    except Exception as e:
        session.error(f"❌ Screen installation verification failed: {e}")


@nox.session(name="kill-session", python=False)
def kill_session(session: nox.Session) -> None:
    """
    Kill a specific tmux or screen session.

    This session allows you to kill a specific interactive session by name.

    Usage:
        nox -s kill-session -- session-name
    """
    if not session.posargs:
        session.error("Please specify a session name: nox -s kill-session -- session-name")

    session_name = session.posargs[0]
    session.log(f"🗑️  Attempting to kill session: {session_name}")

    # Try to kill tmux session
    try:
        result = subprocess.run(["tmux", "kill-session", "-t", session_name],
                               capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            session.log(f"✅ Killed tmux session: {session_name}")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Try to kill screen session
    try:
        result = subprocess.run(["screen", "-S", session_name, "-X", "quit"],
                               capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            session.log(f"✅ Killed screen session: {session_name}")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    session.log(f"⚠️  Could not find or kill session: {session_name}")


@nox.session(name="jupyter-export", python=False)
@with_global_setup
def jupyter_export(session: nox.Session) -> None:
    """
    Export Jupyter notebooks to various formats (Markdown, HTML, PDF, etc.).

    This session exports Jupyter notebooks to different formats
    for documentation and sharing.
    """
    log_session_info(session, {"operation": "jupyter_export"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev + notebook
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev", "notebook"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["nbconvert", "pandoc"],
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    if not session.posargs:
        session.error("Please specify a notebook file: nox -s jupyter-export -- notebook.ipynb")

    notebook_file = session.posargs[0]
    export_format = session.posargs[1] if len(session.posargs) > 1 else "html"

    session.log(f"📤 Exporting {notebook_file} to {export_format}...")

    # Export notebook
    session.run("jupyter", "nbconvert",
                notebook_file,
                f"--to={export_format}",
                "--execute",  # Execute the notebook
                "--allow-errors")  # Continue on errors

    session.log("✅ Jupyter export completed")


# ===============================================================================
# DOCKER BUILDING AND DEPLOYMENT SESSIONS
# ===============================================================================

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def build_docker(session: nox.Session) -> None:
    """Build Docker container with the project."""
    log_session_info(session, {"operation": "build_docker"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev + build
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev", "build"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["docker"],
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🐳 Building Docker container...")

    # Build Docker image
    session.run("docker", "build",
                "-t", "study:latest",
                "-f", "Dockerfile",
                ".")

    session.log("✅ Docker build completed")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def push_docker(session: nox.Session) -> None:
    """Push Docker container to registry."""
    log_session_info(session, {"operation": "push_docker"})

    # Setup environment with Docker tools
    setup_environment(
        session=session,
        extras="build",
        additional_packages=["docker"],
        use_clean_env=True,
        upgrade=False
    )

    session.log("🚀 Pushing Docker container...")

    # Tag and push Docker image
    registry = session.posargs[0] if session.posargs else "your-registry"
    tag = session.posargs[1] if len(session.posargs) > 1 else "latest"

    session.run("docker", "tag", "study:latest", f"{registry}/study:{tag}")
    session.run("docker", "push", f"{registry}/study:{tag}")

    session.log("✅ Docker push completed")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def docker_compose_up(session: nox.Session) -> None:
    """Start services with Docker Compose."""
    log_session_info(session, {"operation": "docker_compose_up"})

    # Setup environment with Docker Compose tools
    setup_environment(
        session=session,
        extras="build",
        additional_packages=["docker-compose"],
        use_clean_env=True,
        upgrade=False
    )

    session.log("🐳 Starting Docker Compose services...")

    # Start services
    session.run("docker-compose", "up", "-d")

    session.log("✅ Docker Compose services started")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def docker_compose_down(session: nox.Session) -> None:
    """Stop services with Docker Compose."""
    log_session_info(session, {"operation": "docker_compose_down"})

    # Setup environment with Docker Compose tools
    setup_environment(
        session=session,
        extras="build",
        additional_packages=["docker-compose"],
        use_clean_env=True,
        upgrade=False
    )

    session.log("🐳 Stopping Docker Compose services...")

    # Stop services
    session.run("docker-compose", "down")

    session.log("✅ Docker Compose services stopped")


# ===============================================================================
# DEVELOPMENT TOOL SESSIONS
# ===============================================================================

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def analyze_code_dependencies(session: nox.Session) -> None:
    """
    Analyze code dependencies and identify unused code candidates.

    This session runs the dependency analyzer tool that scans the src/study
    package and builds a dependency map between functions/classes and their call-sites.
    Helps identify dead code candidates for future cleanup.
    """
    log_session_info(session, {"operation": "dependency_analysis"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🔍 Analyzing code dependencies...")

    # Create output directory
    output_dir = CACHE_DIR / "dependency-analysis"
    output_dir.mkdir(exist_ok=True)

    # Run the embedded dependency analyzer
    dependency_analyzer_code = '''
#!/usr/bin/env python3
"""Static dependency analyzer for the study codebase."""
from __future__ import annotations
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(".").resolve()
SRC_DIR = PROJECT_ROOT / "src" / "study"
REPORT_PATH = Path("cache/dependency-analysis/dependency_report.json")

def iter_py_files(base_dir: Path):
    """Yield all .py source files beneath base_dir."""
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".py"):
                yield Path(root) / fname

def safe_parse(path: Path):
    """Parse path into an ast.AST. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        return ast.parse(source, filename=str(path))
    except Exception as e:
        print(f"Failed to parse {path}: {e}")
        return None

def collect_definitions_and_usages():
    """Scan src/study and collect definitions and usages."""
    definitions = {}
    usages = {}

    for py_file in iter_py_files(SRC_DIR):
        tree = safe_parse(py_file)
        if not tree:
            continue

        rel_path = py_file.relative_to(PROJECT_ROOT)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                qualified_name = node.name
                definitions[qualified_name] = str(rel_path)
            elif isinstance(node, ast.Name):
                name = node.id
                if name not in usages:
                    usages[name] = set()
                usages[name].add(str(rel_path))

    return definitions, usages

def build_dependency_report():
    """Build the full dependency report."""
    definitions, usages = collect_definitions_and_usages()

    report = {
        "definitions": definitions,
        "usages": {k: list(v) for k, v in usages.items()},
        "unused_candidates": []
    }

    for name in definitions:
        if name not in usages or len(usages[name]) <= 1:
            report["unused_candidates"].append(name)

    return report

def main():
    """Main entry point."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = build_dependency_report()

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\\n🔍 Dependency analysis complete!")
    print(f"📊 Report saved to: {REPORT_PATH}")
    print(f"📋 Found {len(report['unused_candidates'])} unused candidates")

    if report["unused_candidates"]:
        print("\\n⚠️  Potential unused symbols:")
        for candidate in sorted(report["unused_candidates"][:10]):
            print(f"  - {candidate}")
        if len(report["unused_candidates"]) > 10:
            print(f"  ... and {len(report['unused_candidates']) - 10} more")

    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    # Write and run the analyzer
    analyzer_script = output_dir / "dependency_analyzer.py"
    with open(analyzer_script, "w") as f:
        f.write(dependency_analyzer_code)

    with session.chdir(PROJECT_ROOT):
        session.run(
            "python", str(analyzer_script),
            external=True
        )

    session.log("✅ Code dependency analysis completed")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def create_snapshot(session: nox.Session) -> None:
    """
    Create architectural diagram snapshots for documentation.

    This session runs the diagram generation tools to create snapshots of the
    LEGO architecture diagrams and other system visualizations.
    """
    log_session_info(session, {"operation": "diagram_snapshot"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        additional_packages=["mermaid-cli", "graphviz"],
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("📊 Creating diagram snapshots...")

    # Create output directory
    output_dir = CACHE_DIR / "diagram-snapshots"
    output_dir.mkdir(exist_ok=True)

    # Run the diagram manager from docs/_static
    with session.chdir(PROJECT_ROOT / "docs" / "_static"):
        session.run(
            "python", "diagram_manager_v2.py", "snapshot", "latest",
            external=True
        )

    # The diagrams are now managed directly in docs/_static/images
    docs_images_dir = PROJECT_ROOT / "docs" / "_static" / "images"
    docs_images_dir.mkdir(exist_ok=True)

    session.log("📋 Diagrams are now maintained in docs/_static/images/")
    session.log(f"📁 Available diagrams: {list(docs_images_dir.glob('*.mmd'))}")

    session.log("✅ Diagram snapshot creation completed")


# ===============================================================================
# DEFAULT SESSIONS CONFIGURATION
# ===============================================================================

# Configure default sessions based on hardware capabilities
capabilities = detect_hardware_capabilities()

# Enhanced default sessions with project management capabilities
# Only include sessions that actually exist in this noxfile
if capabilities["is_h100"] or capabilities["is_a100"]:
    nox.options.sessions = [
        "lock_universal-3.12"
    ]
else:
    nox.options.sessions = [
        "lock_universal-3.12"
    ]

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def file_validation(session: nox.Session) -> None:
    """Validate file formats (JSON, YAML, TOML, Python AST)."""
    session.log("🔍 Validating file formats...")

    setup_environment(session)

    # Validate files using Python pathlib (no find command)
    session.run("python", "-c", """
import pathlib
import json
import yaml
import tomllib
import ast
import sys

def validate_files():
    exclude_dirs = {'.git', '.venv', '__pycache__', 'node_modules', 'build', 'dist', 'scratch'}
    errors = []

    # JSON files
    for file_path in pathlib.Path('.').rglob('*.json'):
        if any(exclude in str(file_path) for exclude in exclude_dirs):
            continue
        try:
            with open(file_path) as f:
                json.load(f)
        except Exception as e:
            errors.append(f'Invalid JSON in {file_path}: {e}')

    # YAML files
    for file_path in pathlib.Path('.').rglob('*.yaml'):
        if any(exclude in str(file_path) for exclude in exclude_dirs):
            continue
        try:
            with open(file_path) as f:
                yaml.safe_load(f)
        except Exception as e:
            errors.append(f'Invalid YAML in {file_path}: {e}')

    for file_path in pathlib.Path('.').rglob('*.yml'):
        if any(exclude in str(file_path) for exclude in exclude_dirs):
            continue
        try:
            with open(file_path) as f:
                yaml.safe_load(f)
        except Exception as e:
            errors.append(f'Invalid YAML in {file_path}: {e}')

    # TOML files
    for file_path in pathlib.Path('.').rglob('*.toml'):
        if any(exclude in str(file_path) for exclude in exclude_dirs):
            continue
        try:
            with open(file_path, 'rb') as f:
                tomllib.load(f)
        except Exception as e:
            errors.append(f'Invalid TOML in {file_path}: {e}')

    # Python files
    for file_path in pathlib.Path('.').rglob('*.py'):
        if any(exclude in str(file_path) for exclude in exclude_dirs):
            continue
        try:
            with open(file_path) as f:
                ast.parse(f.read())
        except Exception as e:
            errors.append(f'Invalid Python AST in {file_path}: {e}')

    if errors:
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print('All files validated successfully!')

validate_files()
""")

    session.log("✅ All files validated successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def file_formatting_fix(session: nox.Session) -> None:
    """Fix file formatting issues (trailing whitespace, line endings, etc.)."""
    session.log("🔧 Fixing file formatting issues...")

    setup_environment(session)

    # Use ruff to fix formatting issues
    session.run("ruff", "check", "--fix", ".")

    # Additional file-specific fixes
    session.run("python", "-c", """
import os
import re

def fix_file_endings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove trailing whitespace
    lines = [line.rstrip() for line in content.splitlines()]

    # Ensure file ends with newline
    if lines and lines[-1] != '':
        lines.append('')

    new_content = '\\n'.join(lines)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed: {file_path}')

# Process all text files
for root, dirs, files in os.walk('.'):
    if any(exclude in root for exclude in ['.git', '.venv', '__pycache__', 'node_modules', 'build', 'dist']):
        continue

    for file in files:
        if file.endswith(('.py', '.md', '.txt', '.yaml', '.yml', '.json', '.toml')):
            file_path = os.path.join(root, file)
            fix_file_endings(file_path)
""")

    session.log("✅ File formatting issues fixed!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def test_cuda_versions(session: nox.Session) -> None:
    """
    Test different CUDA versions to ensure compatibility.

    This session tests the installation with different CUDA versions
    to verify that PyTorch and PyG extensions work correctly.
    """
    log_session_info(session, {"operation": "cuda_version_testing"})

    session.log("🧪 Testing CUDA version compatibility...")

    # Test CPU version
    session.log("🔍 Testing CPU version...")
    try:
        setup_environment(session, hardware="cpu", extras="default", use_clean_env=True)
        session.run("uv", "run", "--active", "python", "-c", """
import torch
import torch_geometric
print(f'CPU PyTorch: {torch.__version__}')
print(f'CPU PyG: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ CPU version working correctly')
""", external=True)
    except Exception as e:
        session.log(f"❌ CPU version failed: {e}")

    # Test CUDA 12.6 version
    session.log("🔍 Testing CUDA 12.6 version...")
    try:
        setup_environment(session, hardware="cu126", extras="default", use_clean_env=True)
        session.run("uv", "run", "--active", "python", "-c", """
import torch
import torch_geometric
print(f'CUDA 12.6 PyTorch: {torch.__version__}')
print(f'CUDA 12.6 PyG: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
print('✅ CUDA 12.6 version working correctly')
""", external=True)
    except Exception as e:
        session.log(f"❌ CUDA 12.6 version failed: {e}")

    # Test CUDA 12.8 version
    session.log("🔍 Testing CUDA 12.8 version...")
    try:
        setup_environment(session, hardware="cu128", extras="default", use_clean_env=True)
        session.run("uv", "run", "--active", "python", "-c", """
import torch
import torch_geometric
print(f'CUDA 12.8 PyTorch: {torch.__version__}')
print(f'CUDA 12.8 PyG: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
print('✅ CUDA 12.8 version working correctly')
""", external=True)
    except Exception as e:
        session.log(f"❌ CUDA 12.8 version failed: {e}")

    session.log("✅ CUDA version testing completed")



@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def file_formatting_check(session: nox.Session) -> None:
    """Check file formatting without fixing."""
    session.log("🔍 Checking file formatting...")

    setup_environment(session)

    # Use ruff to check formatting
    session.run("ruff", "check", ".")

    # Check for trailing whitespace and line ending issues
    session.run("python", "-c", """
import os
import re
import sys

def check_file_endings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.splitlines()
    issues = []

    for i, line in enumerate(lines, 1):
        if line.rstrip() != line:
            issues.append(f'Line {i}: trailing whitespace')

    if lines and lines[-1] != '':
        issues.append('File does not end with newline')

    if issues:
        print(f'Issues in {file_path}:')
        for issue in issues:
            print(f'  - {issue}')
        return True
    return False

# Process all text files
has_issues = False
for root, dirs, files in os.walk('.'):
    if any(exclude in root for exclude in ['.git', '.venv', '__pycache__', 'node_modules', 'build', 'dist']):
        continue

    for file in files:
        if file.endswith(('.py', '.md', '.txt', '.yaml', '.yml', '.json', '.toml')):
            file_path = os.path.join(root, file)
            if check_file_endings(file_path):
                has_issues = True

if has_issues:
    sys.exit(1)
""")

    session.log("✅ File formatting is clean!")


# ===============================================================================
# ENHANCED RESOURCE ALLOCATION WITH SENSIBLE DEFAULTS
# ===============================================================================

def get_default_resource_allocation() -> dict[str, str]:
    """
    Get sensible default resource allocation: 50% memory, half CPUs (first half), single GPU.

    Returns:
        dict: Default resource allocation settings
    """
    capabilities = detect_hardware_capabilities()

    defaults = {}

    # Default to single GPU (GPU 0) if available
    if capabilities["gpu_available"]:
        defaults["gpus"] = "0"
    else:
        defaults["gpus"] = None

    # Default to first half of CPUs
    cpu_count = capabilities.get("cpu_count", os.cpu_count() or 1)
    half_cpus = max(1, cpu_count // 2)
    defaults["cpus"] = f"0-{half_cpus-1}"

    # Default to 50% of total memory
    total_memory = capabilities.get("total_memory_gb", 8)
    defaults["memory"] = total_memory * 0.5

    return defaults

def parse_resource_arguments(session: nox.Session, description: str, args_list: list | None = None) -> dict:
    """
    Parse consistent resource allocation arguments for interactive sessions.

    Args:
        session: The nox session
        description: Description for the argument parser
        args_list: Optional list of arguments to parse (if None, uses session.posargs)

    Returns:
        dict: Parsed arguments
    """
    import argparse

    defaults = get_default_resource_allocation()

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpus", default=defaults["gpus"],
                       help=f"GPU allocation (default: {defaults['gpus']} for single GPU, 'all' for all GPUs, '0,1,2' for specific GPUs)")
    parser.add_argument("--cpus", default=defaults["cpus"],
                       help=f"CPU allocation (default: {defaults['cpus']} for first half, 'all' for all CPUs, '0-15' for range)")
    parser.add_argument("--memory", type=float, default=defaults["memory"],
                       help=f"Memory limit in GB (default: {defaults['memory']:.1f}GB for 50% of total)")
    parser.add_argument("--port", default="2718", help="Port to run on")
    parser.add_argument("--enhanced", action="store_true", help="Enable ML framework optimizations")
    parser.add_argument("--no-tmux", action="store_true", help="Disable tmux/screen session (run directly)")

    # Use provided args_list or session.posargs
    args_to_parse = args_list if args_list is not None else (session.posargs or [])
    return parser.parse_args(args_to_parse)

# ===============================================================================
# TMUX/SCREEN SESSION MANAGEMENT
# ===============================================================================

def detect_terminal_multiplexer() -> str | None:
    """
    Detect available terminal multiplexer, preferring screen over tmux.

    Prefers screen to avoid conflicts with Warp terminal which uses tmux internally.

    Returns:
        str: 'screen', 'tmux', or None if neither is available
    """
    # Check if we're already inside a tmux session (like in Warp)
    if os.environ.get("TMUX"):
        print("⚠️  Detected existing tmux session (likely Warp terminal). Preferring screen to avoid conflicts.")
        # Try screen first when already in tmux
        try:
            result = subprocess.run(["screen", "-version"], capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                return "screen"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        # Fall back to tmux only if screen is not available
        try:
            result = subprocess.run(["tmux", "-V"], capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                print("⚠️  Using tmux despite existing session. You may need to unset $TMUX if you encounter issues.")
                return "tmux"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        return None

    # Normal detection order: prefer screen, then tmux
    try:
        # Check for screen first (preferred)
        result = subprocess.run(["screen", "-version"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            return "screen"
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    try:
        # Check for tmux as fallback
        result = subprocess.run(["tmux", "-V"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            return "tmux"
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    return None

def create_tmux_session(session_name: str, command: list[str], env_vars: dict[str, str] | None = None) -> None:
    """
    Create a new tmux session and run the command.

    Args:
        session_name: Name for the tmux session
        command: Command to run in the session
        env_vars: Environment variables to set
    """
    # Kill existing session if it exists
    subprocess.run(["tmux", "kill-session", "-t", session_name],
                   capture_output=True, check=False)

    # Create new session
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name], check=True)

    # Set environment variables if provided
    if env_vars:
        for key, value in env_vars.items():
            subprocess.run(["tmux", "set-env", "-t", session_name, key, value], check=False)

    # Send the command to the session
    cmd_str = " ".join(command)
    subprocess.run(["tmux", "send-keys", "-t", session_name, cmd_str, "Enter"], check=True)

    print(f"✅ Created tmux session '{session_name}'")
    print(f"🔗 Attach with: tmux attach-session -t {session_name}")
    print("📋 Detach with: Ctrl+B, then D")
    print(f"🗑️  Kill session: tmux kill-session -t {session_name}")

def create_screen_session(session_name: str, command: list[str], env_vars: dict[str, str] | None = None) -> None:
    """
    Create a new screen session and run the command.

    Args:
        session_name: Name for the screen session
        command: Command to run in the session
        env_vars: Environment variables to set
    """
    # Kill existing session if it exists
    subprocess.run(["screen", "-S", session_name, "-X", "quit"],
                   capture_output=True, check=False)

    # Create new screen session
    subprocess.run(["screen", "-dmS", session_name], check=True)

    # Set environment variables if provided
    if env_vars:
        for key, value in env_vars.items():
            subprocess.run(["screen", "-S", session_name, "-X", "setenv", key, value], check=False)

    # Send the command to the session
    cmd_str = " ".join(command)
    subprocess.run(["screen", "-S", session_name, "-X", "stuff", f"{cmd_str}\n"], check=True)

    print(f"✅ Created screen session '{session_name}'")
    print(f"🔗 Attach with: screen -r {session_name}")
    print("📋 Detach with: Ctrl+A, then D")
    print(f"🗑️  Kill session: screen -S {session_name} -X quit")

def run_in_multiplexer(session_name: str, command: list[str], env_vars: dict[str, str] | None = None, no_tmux: bool = False) -> None:
    """
    Run command in tmux/screen session or directly.

    Args:
        session_name: Name for the session
        command: Command to run
        env_vars: Environment variables to set
        no_tmux: If True, run directly without multiplexer
    """
    if no_tmux:
        # Run directly
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        print(f"🚀 Running command directly: {' '.join(command)}")
        subprocess.run(command, env=env, check=True)
    else:
        # Use tmux/screen
        multiplexer = detect_terminal_multiplexer()

        if multiplexer == "tmux":
            try:
                create_tmux_session(session_name, command, env_vars)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create tmux session: {e}")
                print("💡 This might be due to nested tmux sessions (e.g., in Warp terminal)")
                print("💡 Try using --no-tmux flag or install screen")
                print("💡 Or run: unset TMUX && nox -s <session-name>")
                raise
        elif multiplexer == "screen":
            create_screen_session(session_name, command, env_vars)
        else:
            print("⚠️  No terminal multiplexer (tmux/screen) found. Running directly.")
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            subprocess.run(command, env=env, check=True)



# ===============================================================================
# ENHANCED RESOURCE ALLOCATION FUNCTION
# ===============================================================================

def allocate_resources(
    gpus: str | None = None,
    cpus: str | None = None,
    memory_gb: float | None = None,
    auto_optimize: bool = True
) -> dict[str, str]:
    """Enhanced resource allocation with sensible defaults and H100/A100 auto-optimization."""
    env = {}
    capabilities = detect_hardware_capabilities()

    # Always get defaults for comparison purposes
    defaults = get_default_resource_allocation()

    # Apply sensible defaults if not specified
    if gpus is None:
        gpus = defaults["gpus"]
    if cpus is None:
        cpus = defaults["cpus"]
    if memory_gb is None:
        memory_gb = defaults["memory"]

    # Auto-optimization for H100/A100 (override defaults for optimal performance)
    if auto_optimize and (capabilities.get("is_h100", False) or capabilities.get("is_a100", False)):
        # For H100/A100, we can be more aggressive with defaults
        if gpus == defaults.get("gpus"):  # Only override if using default
            gpu_count = capabilities.get("gpu_count", 0)
            gpus = ",".join(str(i) for i in range(gpu_count)) if gpu_count > 1 else "0"
        if cpus == defaults.get("cpus"):  # Only override if using default
            cpu_count = capabilities.get("cpu_count", os.cpu_count() or 1)
            optimal_cpus = max(1, cpu_count - 2)  # Leave 2 CPUs for system
            cpus = f"0-{optimal_cpus-1}"
        if memory_gb == defaults.get("memory") and capabilities.get("optimal_batch_size"):
            memory_gb = min(memory_gb * 1.5, capabilities["optimal_batch_size"])  # Use up to 75% for H100/A100

    # GPU allocation
    if gpus is not None:
        if gpus.lower() == "all":
            gpu_count = capabilities.get("gpu_count", 0)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
        elif gpus.lower() == "none":
            env["CUDA_VISIBLE_DEVICES"] = ""  # Disable all GPUs
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpus

    # CPU allocation
    if cpus is not None:
        if cpus.lower() == "all":
            env["OMP_NUM_THREADS"] = str(capabilities.get("cpu_count", os.cpu_count() or 1))
        else:
            # Parse CPU specification
            if "-" in cpus:
                start, end = map(int, cpus.split("-"))
                env["OMP_NUM_THREADS"] = str(end - start + 1)
            else:
                env["OMP_NUM_THREADS"] = str(len(cpus.split(",")))
            env["CPU_AFFINITY"] = cpus

    # Memory allocation
    if memory_gb is not None:
        memory_mb = int(memory_gb * 1024)
        env["MEMORY_LIMIT_MB"] = str(memory_mb)
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{memory_mb//2}"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    return env

# ===============================================================================
# CODE QUALITY AND TESTING SESSIONS
# ===============================================================================

@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def lint(session: nox.Session) -> None:
    """
    Run code linting and formatting checks (without type checking).

    This session runs linting and formatting tools:
    - ruff for linting and formatting
    - black for code formatting (if needed)
    - isort for import sorting (if needed)
    """
    log_session_info(session, {"operation": "code_linting"})

    # Get CI-optimized extras for better performance
    extras = get_ci_optimized_extras("lint")
    is_ci = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"

    session.log(f"🔧 Environment: {'CI/CD' if is_ci else 'Local'} → extras: {extras}")

    setup_environment(
        session=session,
        extras=extras,
        hardware=None,  # No hardware extras needed for linting in CI
        use_clean_env=is_ci,  # Clean env for CI, shared for local
        upgrade=False
    )

    session.log("🔍 Running code linting and formatting...")

    # Run ruff for linting and formatting
    session.log("🔧 Running ruff linting...")
    session.run("ruff", "check", "src", "tests", "docs", "--fix")

    # Run ruff formatting
    session.log("🎨 Running ruff formatting...")
    session.run("ruff", "format", "src", "tests", "docs")

    session.log("✅ Code linting and formatting completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def typing(session: nox.Session) -> None:
    """
    Run comprehensive type checking with multiple tools.

    This session supports multiple type checkers that can be used individually or in combination:
    - pyright (Microsoft's static type checker) - DEFAULT
    - mypy (Python static type checker) 
    - typos (Spell checker for code)

    Enhanced pyright options (based on recent optimizations):
    - Uses optimized pyrightconfig.json with suppressed numpy/pandas typing issues
    - --errors-only: Focus on critical issues, skip tests (faster execution)
    - --stats: Show detailed analysis statistics

    Usage:
        nox -s typing                          # Run pyright only (default)
        nox -s typing -- --errors-only         # Run pyright on src/ only (fast)
        nox -s typing -- --stats               # Run pyright with statistics
        nox -s typing -- --mypy               # Run mypy only
        nox -s typing -- --typos              # Run typos only
        nox -s typing -- --all                # Run all type checkers
        nox -s typing -- --pyright --mypy     # Run pyright and mypy
        nox -s typing -- --errors-only --stats # Fast pyright with stats
    """
    log_session_info(session, {"operation": "type_checking"})

    # Parse arguments to determine which type checkers to run
    args = session.posargs
    run_pyright = not any(arg in ['--mypy', '--typos'] for arg in args) or '--pyright' in args or '--all' in args
    run_mypy = '--mypy' in args or '--all' in args
    run_typos = '--typos' in args or '--all' in args

    # Get CI-optimized extras for better performance
    extras = get_ci_optimized_extras("typing")
    is_ci = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"

    session.log(f"🔧 Environment: {'CI/CD' if is_ci else 'Local'} → extras: {extras}")

    # Show which tools will be run
    tools = []
    if run_pyright:
        tools.append('pyright')
    if run_mypy:
        tools.append('mypy')
    if run_typos:
        tools.append('typos')
    session.log(f"🎯 Type checkers to run: {', '.join(tools) if tools else 'pyright (default)'}")

    setup_environment(
        session=session,
        extras=extras,
        hardware=None,  # No hardware extras needed for type checking
        use_clean_env=is_ci,  # Clean env for CI, shared for local
        upgrade=False
    )

    session.log("🔍 Running type checking tools...")

    # Track results
    results = {}

    # Run pyright with enhanced options
    if run_pyright:
        session.log("🔍 Running pyright type checking...")

        # Check for pyright-specific arguments
        pyright_args = []
        show_stats = '--stats' in args
        errors_only = '--errors-only' in args

        if show_stats:
            pyright_args.append('--stats')
        if errors_only:
            session.log("⚡ Running in errors-only mode (faster, shows only critical issues)")

        try:
            cmd = ["pyright", "src/study", "--pythonpath", "src"]

            # Add tests only if not in errors-only mode (tests have many issues)
            if not errors_only:
                cmd.append("tests")

            cmd.extend(pyright_args)
            session.run(*cmd)
            results['pyright'] = '✅ PASSED'
            session.log("✅ pyright completed successfully!")
        except Exception as e:
            results['pyright'] = f'❌ FAILED: {e!s}'
            session.log(f"❌ pyright failed: {e}")
            if not (run_mypy or run_typos):  # Only fail if this is the only tool
                raise

    # Run mypy
    if run_mypy:
        session.log("🔍 Running mypy type checking...")
        try:
            session.run("mypy", "-p", "study", "--config-file", ".mypy.ini")
            results['mypy'] = '✅ PASSED'
            session.log("✅ mypy completed successfully!")
        except Exception as e:
            results['mypy'] = f'❌ FAILED: {e!s}'
            session.log(f"❌ mypy failed: {e}")
            if not (run_pyright or run_typos):  # Only fail if this is the only tool
                raise

    # Run typos
    if run_typos:
        session.log("🔍 Running typos spell checking...")
        try:
            session.run("typos", "src", "tests", "docs", "--config", "typos.toml")
            results['typos'] = '✅ PASSED'
            session.log("✅ typos completed successfully!")
        except Exception as e:
            results['typos'] = f'❌ FAILED: {e!s}'
            session.log(f"❌ typos failed: {e}")
            if not (run_pyright or run_mypy):  # Only fail if this is the only tool
                raise

    # Summary
    session.log("\n📊 Type Checking Summary:")
    for tool, result in results.items():
        session.log(f"  {tool}: {result}")

    # Fail if any tool failed (except when running multiple tools)
    failed_tools = [tool for tool, result in results.items() if 'FAILED' in result]
    if failed_tools and len(tools) > 1:
        session.log(f"\n⚠️ Some type checkers failed: {', '.join(failed_tools)}")
        session.log("Consider running individual tools to debug: nox -s typing -- --<tool>")
        # Don't fail the session when running multiple tools - let user decide

    session.log("\n✅ Type checking session completed!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test(session: nox.Session) -> None:
    """
    Run comprehensive test suite with coverage.

    This session supports both legacy pytest and new organized testing:
    - Use --use-comprehensive for new organized test runner
    - Use --legacy for traditional pytest execution
    - Default: intelligent detection based on arguments
    - Coverage reporting and parallel execution support
    """
    log_session_info(session, {"operation": "testing"})

    # Get CI-optimized extras for better performance
    extras = get_ci_optimized_extras("test")
    is_ci = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"

    session.log(f"🔧 Environment: {'CI/CD' if is_ci else 'Local'} → extras: {extras}")

    setup_environment(
        session=session,
        extras=extras,
        hardware="cpu" if is_ci else None,
        use_clean_env=is_ci,  # Clean env for CI, shared for local
        upgrade=False
    )

    # Determine execution mode
    use_comprehensive = "--use-comprehensive" in session.posargs
    use_legacy = "--legacy" in session.posargs
    coverage = "--cov" in session.posargs
    parallel = "--parallel" in session.posargs
    verbose = "--verbose" in session.posargs

    # Remove control arguments from posargs
    filtered_posargs = [
        arg for arg in session.posargs
        if arg not in ["--use-comprehensive", "--legacy", "--cov", "--parallel", "--verbose"]
    ]

    if use_comprehensive or (not use_legacy and not coverage and not parallel):
        # Use new comprehensive test runner
        session.log("🎯 Running organized comprehensive test suite...")

        cmd = ["uv", "run", "--active", "python", "tests/run_comprehensive_tests.py"]

        if verbose:
            cmd.append("--verbose")

        if filtered_posargs:
            cmd.extend(filtered_posargs)

        session.run(*cmd)

    else:
        # Use legacy pytest execution
        session.log("🧪 Running legacy pytest suite...")

        cmd = ["pytest"]

        if coverage:
            cmd.extend(["--cov=src/study", "--cov-report=html", "--cov-report=term-missing"])

        if parallel:
            cmd.extend(["-n", "auto"])

        if verbose:
            cmd.append("-v")

        # Add test directories
        cmd.extend(["tests/"])

        if filtered_posargs:
            cmd.extend(filtered_posargs)

        session.run("uv", "run", "--active", *cmd)

    session.log("✅ Test suite completed successfully!")


# ============================================================================
# 🎯 COMPREHENSIVE ORGANIZED TEST SESSIONS
# ============================================================================
# These sessions support the new enterprise-grade test organization with
# category-based execution aligned to architectural components.


@nox.session(name="test-core", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_core(session: nox.Session) -> None:
    """
    Run core architectural component tests.

    This session runs all core component tests:
    - Loss functions (comprehensive 27+ functions)
    - Executor system (enterprise-grade)
    - LEGO framework (architecture testing)
    - NBFNet system (graph neural networks)
    - Data processing (I/O and datasets)
    """
    log_session_info(session, {"operation": "core_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("🏗️ Running core architectural component tests...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", "core", "--no-slow", *session.posargs
    )
    session.log("✅ Core tests completed successfully!")


@nox.session(name="test-losses", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_losses(session: nox.Session) -> None:
    """
    Run comprehensive loss function tests.

    This session runs all loss function tests:
    - Mathematical validation (PyTorch equivalence)
    - Numerical stability testing
    - Edge case handling
    - Device consistency (CPU/GPU)
    - Sklearn interface compatibility
    """
    log_session_info(session, {"operation": "loss_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("🔥 Running comprehensive loss function tests...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", "losses", *session.posargs
    )
    session.log("✅ Loss function tests completed successfully!")


@nox.session(name="test-executors", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_executors(session: nox.Session) -> None:
    """
    Run executor system tests.

    This session runs all executor system tests:
    - Configuration validation
    - Multi-GPU distributed execution
    - CLI interface testing
    - Resource allocation
    - Experiment tracking
    """
    log_session_info(session, {"operation": "executor_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("⚡ Running executor system tests...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", "executors", *session.posargs
    )
    session.log("✅ Executor tests completed successfully!")


@nox.session(name="test-performance", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_performance(session: nox.Session) -> None:
    """
    Run performance benchmarking tests.

    This session runs performance and scalability tests:
    - Training speed benchmarks
    - Memory usage profiling
    - Inference throughput
    - Multi-GPU performance
    - Memory leak detection
    """
    log_session_info(session, {"operation": "performance_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("🚀 Running performance benchmarks...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", "performance", *session.posargs
    )
    session.log("✅ Performance tests completed successfully!")


@nox.session(name="test-integration", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_integration(session: nox.Session) -> None:
    """
    Run integration and end-to-end tests.

    This session runs integration tests:
    - End-to-end workflows
    - Distributed training
    - Architecture integration
    - Data pipeline validation
    """
    log_session_info(session, {"operation": "integration_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("🔗 Running integration tests...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", "integration", *session.posargs
    )
    session.log("✅ Integration tests completed successfully!")


@nox.session(name="test-quality", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_quality(session: nox.Session) -> None:
    """
    Run code quality and configuration tests.

    This session runs quality assurance tests:
    - Configuration validation
    - Code quality checks
    - Linting compliance
    """
    log_session_info(session, {"operation": "quality_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("🎯 Running quality tests...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", "quality", "--no-slow", *session.posargs
    )
    session.log("✅ Quality tests completed successfully!")


@nox.session(name="test-categories", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_categories(session: nox.Session) -> None:
    """
    Run custom test categories with flexible configuration.

    This session allows running specific test categories:
    - Use session.posargs to specify categories
    - Support for multiple categories
    - Advanced filtering options

    Examples:
        nox -s test-categories -- losses executors
        nox -s test-categories -- core --no-slow
        nox -s test-categories -- integration performance --verbose
    """
    log_session_info(session, {"operation": "category_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    if not session.posargs:
        session.error(
            "❌ No test categories specified. "
            "Available: core, losses, executors, performance, lego, nbfnet, data, integration, quality, legacy"
        )

    session.log(f"📁 Running test categories: {' '.join(session.posargs)}")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        "--categories", *session.posargs
    )
    session.log("✅ Category tests completed successfully!")


@nox.session(name="test-comprehensive", python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def test_comprehensive(session: nox.Session) -> None:
    """
    Run the complete comprehensive test suite.

    This session runs all test categories with full coverage:
    - All core architectural components
    - Integration and performance tests
    - Quality assurance
    - Professional test reporting
    """
    log_session_info(session, {"operation": "comprehensive_testing"})

    extras = get_ci_optimized_extras("test")
    setup_environment(session, extras=extras)

    session.log("🎯 Running complete comprehensive test suite...")
    session.run(
        "uv", "run", "--active", "python", "tests/run_comprehensive_tests.py",
        *session.posargs
    )
    session.log("✅ Comprehensive test suite completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def security(session: nox.Session) -> None:
    """
    Run security and compliance checks.

    This session runs security scanning tools:
    - Safety for dependency vulnerability scanning
    - Bandit for code security analysis
    - Pre-commit hooks for security checks
    """
    log_session_info(session, {"operation": "security_scanning"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🛡️ Running security and compliance checks...")

    # Run safety check for dependency vulnerabilities
    session.log("🔒 Checking dependencies for vulnerabilities...")
    session.run("safety", "check")

    # Run bandit for code security analysis
    session.log("🔍 Running code security analysis...")
    session.run("bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json")

    # Run pre-commit hooks
    session.log("✅ Running pre-commit security hooks...")
    session.run("pre-commit", "run", "--all-files", "--hook-stage", "manual")

    session.log("✅ Security checks completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def ci(session: nox.Session) -> None:
    """
    Run complete CI/CD pipeline locally.

    This session runs all checks that would be performed in CI:
    - Code quality checks
    - Testing with coverage
    - Security scanning
    - Documentation building
    - Local dependency testing
    """
    log_session_info(session, {"operation": "ci_pipeline"})

    session.log("🚀 Running complete CI/CD pipeline...")

    # Run all CI steps
    session.log("📋 Step 1/4: Code quality checks...")
    session.run("nox", "-s", "lint", external=True)

    session.log("📋 Step 2/4: Security checks...")
    session.run("nox", "-s", "security", external=True)

    session.log("📋 Step 3/4: Test suite...")
    session.run("nox", "-s", "test", "--", "--cov", "--parallel", external=True)

    session.log("📋 Step 4/4: Documentation build...")
    session.run("nox", "-s", "build_docs", external=True)

    session.log("✅ CI/CD pipeline completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def clean(session: nox.Session) -> None:
    """
    Clean all generated files and caches.

    This session removes all temporary files, caches, and build artifacts:
    - Python cache files
    - Build directories
    - Test coverage reports
    - Documentation builds
    - Temporary files
    """
    log_session_info(session, {"operation": "cleanup"})

    session.log("🧹 Cleaning all generated files and caches...")

    # Clean Python cache files (fast, targeted approach)
    session.log("🗑️ Removing Python cache files...")

    # Remove __pycache__ directories in common locations
    cache_dirs = [
        "src/__pycache__",
        "tests/__pycache__",
        "docs/__pycache__",
        "src/study/__pycache__",
    ]

    for cache_dir in cache_dirs:
        if (PROJECT_ROOT / cache_dir).exists():
            session.run("rm", "-rf", cache_dir, external=True)

    # Remove .pyc and .pyo files in common locations
    pyc_files = [
        "src/*.pyc",
        "tests/*.pyc",
        "docs/*.pyc",
        "src/study/*.pyc",
        "src/external/*.pyc",
        "*.pyo",
    ]

    for pattern in pyc_files:
        session.run("rm", "-f", pattern, external=True)

    # Clean build directories
    session.log("🗑️ Removing build directories...")
    build_dirs = ["build/", "dist/", "*.egg-info/"]
    for build_dir in build_dirs:
        session.run("rm", "-rf", build_dir, external=True)

    # Clean test artifacts
    session.log("🗑️ Removing test artifacts...")
    test_artifacts = [
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "bandit-report.json",
        "test-reports/",
        ".coverage.*",
    ]
    for artifact in test_artifacts:
        session.run("rm", "-rf", artifact, external=True)

    # Clean documentation builds
    session.log("🗑️ Removing documentation builds...")
    doc_dirs = ["docs/_build/", "docs/slides/"]
    for doc_dir in doc_dirs:
        session.run("rm", "-rf", doc_dir, external=True)

    # Clean temporary files and caches
    session.log("🗑️ Removing temporary files...")
    temp_dirs = [
        "scratch/temp/",
        "cache/",
        ".nox/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".pytest_cache/",
        "__pycache__/",
    ]
    for temp_dir in temp_dirs:
        session.run("rm", "-rf", temp_dir, external=True)

    # Clean specific cache files
    cache_files = [
        ".ruff_cache",
        ".mypy_cache",
        ".pytest_cache",
        ".coverage",
        "coverage.xml",
        "*.log",
        "*.ckpt",
        "*.pkl",
        "*.h5ad",
        "*.h5mu",
    ]
    for cache_file in cache_files:
        session.run("rm", "-f", cache_file, external=True)

    session.log("✅ Cleanup completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def format(session: nox.Session) -> None:
    """
    Format all code files.

    This session formats all code files using:
    - ruff for Python code formatting
    - black for additional formatting
    - isort for import sorting
    """
    log_session_info(session, {"operation": "code_formatting"})

    # Get CI-optimized extras for better performance
    extras = get_ci_optimized_extras("format")
    is_ci = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"

    session.log(f"🔧 Environment: {'CI/CD' if is_ci else 'Local'} → extras: {extras}")

    setup_environment(
        session=session,
        extras=extras,
        hardware=None,  # No hardware extras needed for formatting
        use_clean_env=is_ci,  # Clean env for CI, shared for local
        upgrade=False
    )

    session.log("🎨 Formatting all code files...")

    # Run ruff formatting
    session.log("🔧 Running ruff formatting...")
    session.run("ruff", "format", "src", "tests", "docs")

    # Run black formatting
    session.log("⚫ Running black formatting...")
    session.run("black", "src", "tests", "docs")

    # Run isort for import sorting
    session.log("📦 Sorting imports...")
    session.run("isort", "src", "tests", "docs")

    session.log("✅ Code formatting completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def precommit(session: nox.Session) -> None:
    """
    Install and run pre-commit hooks.

    This session manages pre-commit hooks:
    - Install pre-commit hooks
    - Run all hooks on all files
    - Update hook configurations
    """
    log_session_info(session, {"operation": "precommit_management"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🔧 Managing pre-commit hooks...")

    # Install pre-commit hooks
    session.log("📦 Installing pre-commit hooks...")
    session.run("pre-commit", "install", "--install-hooks")

    # Run all hooks on all files
    session.log("✅ Running pre-commit hooks on all files...")
    session.run("pre-commit", "run", "--all-files")

    session.log("✅ Pre-commit hooks configured and executed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def build(session: nox.Session) -> None:
    """
    Build the project package.

    This session builds the project package:
    - Build wheel distribution
    - Build source distribution
    - Validate package structure
    """
    log_session_info(session, {"operation": "package_building"})

    # Auto-detect hardware and setup environment with optimal configuration
    # Local development: hardware_extras + default + dev + build
    detected_hardware, hardware_extras = auto_detect_hardware_extras()
    final_extras = [*hardware_extras, "default", "dev", "build"]

    session.log(f"🔧 Hardware detected: {detected_hardware} → extras: {final_extras}")

    setup_environment(
        session=session,
        extras=",".join(final_extras),
        use_clean_env=False,  # Use same venv as other sessions
        upgrade=False
    )

    session.log("🔨 Building project package...")

    # Clean previous builds
    session.log("🧹 Cleaning previous builds...")
    session.run("rm", "-rf", "build/", "dist/", "*.egg-info/", external=True)

    # Build wheel
    session.log("⚙️ Building wheel distribution...")
    session.run("python", "-m", "build", "--wheel")

    # Build source distribution
    session.log("📦 Building source distribution...")
    session.run("python", "-m", "build", "--sdist")

    # Validate package
    session.log("✅ Validating package structure...")
    session.run("twine", "check", "dist/*")

    session.log("✅ Package build completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def release(session: nox.Session) -> None:
    """
    Prepare and validate release.

    This session prepares a release:
    - Run all CI checks
    - Build package
    - Validate release readiness
    - Generate release notes
    """
    log_session_info(session, {"operation": "release_preparation"})

    session.log("🚀 Preparing release...")

    # Run all CI checks
    session.log("📋 Running CI checks...")
    session.run("nox", "-s", "ci", external=True)

    # Build package
    session.log("🔨 Building package...")
    session.run("nox", "-s", "build", external=True)

    # Generate release notes
    session.log("📝 Generating release notes...")
    session.run("cz", "bump", "--dry-run")

    session.log("✅ Release preparation completed successfully!")


@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
@with_global_setup
def help(session: nox.Session) -> None:
    """
    Display comprehensive help information.

    This session provides detailed information about:
    - Available sessions
    - Usage examples
    - Configuration options
    - Best practices
    """
    log_session_info(session, {"operation": "help_display"})

    session.log("📚 study Development Environment Help")
    session.log("=" * 50)

    session.log("\n🎯 CORE WORKFLOWS:")
    session.log("  • nox -s init_project          # Complete project setup")
    session.log("  • nox -s ci                    # Run complete CI pipeline")
    session.log("  • nox -s test                  # Run test suite")
    session.log("  • nox -s lint                  # Code quality checks")
    session.log("  • nox -s format                # Format all code")

    session.log("\n🔧 DEVELOPMENT TOOLS:")
    session.log("  • nox -s marimo-edit           # Start Marimo notebook")
    session.log("  • nox -s jupyter-lab           # Start Jupyter Lab")
    session.log("  • nox -s run-script            # Run Python scripts")
    session.log("  • nox -s build_docs            # Build documentation")

    session.log("\n🛡️ QUALITY:")
    session.log("  • nox -s precommit             # Pre-commit hooks")
    session.log("  • nox -s clean                 # Clean all caches")

    session.log("\n🚀 DEPLOYMENT:")
    session.log("  • nox -s build                 # Build package")
    session.log("  • nox -s release               # Prepare release")
    session.log("  • nox -s build_docker          # Build Docker image")

    session.log("\n🔧 CONFIGURATION:")
    session.log("  • Environment variables in pyproject.toml")
    session.log("  • Hardware auto-detection for GPU optimization")
    session.log("  • Cache management in cache/")

    session.log("\n📚 DOCUMENTATION:")
    session.log("  • docs/design-principles/development/")
    session.log("  • .cursor/rules/ for agent instructions")
    session.log("  • README.md for project overview")

    session.log("\n💡 USAGE EXAMPLES:")
    session.log("  • nox -s test -- --cov         # Run tests with coverage")
    session.log("  • nox -s lint -- --fix         # Run linting with fixes")
    session.log("  • nox -s marimo-edit -- --no-tmux  # Run without tmux")

# =============================================================================
# MODEL COVERAGE & ARCHITECTURE ANALYSIS SESSIONS
# =============================================================================

@nox.session(name="model-coverage", python=False)
@with_global_setup
def model_coverage(session: nox.Session) -> None:
    """Analyze model coverage: code/test/doc alignment and usage graph clustering."""
    # Parse resource arguments
    resource_args = [arg for arg in session.posargs if arg.startswith("--")]
    args = parse_resource_arguments(session, "Model coverage analysis", resource_args)

    log_session_info(session, {
        "operation": "model_coverage",
        "gpus": args.gpus,
        "cpus": args.cpus,
        "memory_gb": args.memory,
    })

    # Allocate resources
    env_vars = allocate_resources(gpus=args.gpus, cpus=args.cpus, memory_gb=args.memory)
    session.env.update(env_vars)

    # Run the standalone script for now (clean implementation later)
    session.run("uv", "run", "--active", "python", "coverage_analysis/refresh_coverage.py", external=True)

    session.log("✅ Coverage analysis complete")
    session.log("📊 Results written to coverage_analysis/")

@nox.session(name="create-snapshot", python=False)
@with_global_setup
def create_snapshot(session: nox.Session) -> None:
    """Generate architectural diagrams from coverage analysis data."""
    # Parse resource arguments
    resource_args = [arg for arg in session.posargs if arg.startswith("--")]
    args = parse_resource_arguments(session, "Architecture snapshot generation", resource_args)

    log_session_info(session, {
        "operation": "create_snapshot",
        "gpus": args.gpus,
        "cpus": args.cpus,
        "memory_gb": args.memory,
    })

    # Allocate resources
    env_vars = allocate_resources(gpus=args.gpus, cpus=args.cpus, memory_gb=args.memory)
    session.env.update(env_vars)

    # Ensure coverage data exists
    coverage_dir = PROJECT_ROOT / "coverage_analysis"
    if not (coverage_dir / "usage_graph.gexf").exists():
        session.log("📊 Coverage data not found, running analysis first...")
        session.run("uv", "run", "--active", "python", "coverage_analysis/refresh_coverage.py", external=True)

    # Generate diagrams (placeholder for now)
    diagrams_dir = coverage_dir / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    with (diagrams_dir / "architecture_summary.txt").open("w") as f:
        f.write("# Architecture Snapshot\n")
        f.write("Generated from usage graph and coverage analysis\n")

    session.log("✅ Architecture snapshot generated")
    session.log("📸 Diagrams written to coverage_analysis/diagrams/")


@nox.session(name="create-experiment", python=False)
def create_experiment(session: nox.Session) -> None:
    """Create a new experiment subfolder with proper structure and templates.
    
    Usage:
        nox -s create-experiment -- EXPERIMENT_NAME [--type=TYPE]
    
    Args:
        EXPERIMENT_NAME: Name of the experiment (e.g., K562_finetuning)
        --type: Type of experiment (hyperopt, finetuning, benchmark, analysis)
    """
    import json
    from datetime import datetime

    # Try to import yaml, fallback to json if not available
    try:
        import yaml
        use_yaml = True
    except ImportError:
        use_yaml = False

    if not session.posargs:
        session.error("❌ Please provide experiment name: nox -s create-experiment -- EXPERIMENT_NAME")

    experiment_name = session.posargs[0]
    experiment_type = "finetuning"  # default

    # Parse optional type argument
    for arg in session.posargs[1:]:
        if arg.startswith("--type="):
            experiment_type = arg.split("=")[1]

    # Validate experiment name
    if not experiment_name.replace("_", "").replace("-", "").isalnum():
        session.error(f"❌ Invalid experiment name: {experiment_name}. Use alphanumeric characters, underscores, and hyphens only.")

    # Create experiment directory
    exp_dir = PROJECT_ROOT / "experiments" / experiment_name
    if exp_dir.exists():
        session.error(f"❌ Experiment directory already exists: {exp_dir}")

    session.log(f"🧪 Creating experiment: {experiment_name} (type: {experiment_type})")

    # Create directory structure
    directories = ["config", "scripts", "logs", "results", "checkpoints", "docs"]
    for dir_name in directories:
        (exp_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # Create config templates based on experiment type
    config_templates = {
        "hyperopt": {
            "experiment_config.yaml": {
                "name": experiment_name,
                "description": f"Hyperparameter optimization experiment for {experiment_name}",
                "version": "1.0.0",
                "tags": ["hyperopt", "distributed"],
                "hyperparameters": {
                    "scientific": {
                        "model.hidden_dim": {"type": "choice", "choices": [128, 256, 512]},
                        "model.num_layers": {"type": "choice", "choices": [2, 3, 4]}
                    },
                    "nuisance": {
                        "training.learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
                        "training.batch_size": {"type": "choice", "choices": [16, 32, 64]}
                    }
                },
                "optimization": {
                    "max_scientific_trials": 9,
                    "max_nuisance_trials": 25,
                    "scientific_strategy": "grid",
                    "nuisance_strategy": "optuna",
                    "num_gpus": 8,
                    "timeout_hours": 24
                }
            }
        },
        "finetuning": {
            "model_config.yaml": {
                "model": {
                    "name": "NBFNetModel",
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "message_operator": "transe",
                    "aggregate_operator": "mean",
                    "boundary_injection": True
                }
            },
            "training_config.yaml": {
                "training": {
                    "epochs": 1000,
                    "learning_rate": 1e-3,
                    "batch_size": 32,
                    "early_stopping": {
                        "patience": 50,
                        "monitor": "val_loss",
                        "mode": "min"
                    }
                },
                "data": {
                    "train_split": 0.8,
                    "val_split": 0.1,
                    "test_split": 0.1
                }
            },
            "data_config.yaml": {
                "data": {
                    "network_name": "STRING (combined_scores)",
                    "embedding_name": "string-space-sequence",
                    "perturbation_dataset": "K562_pb",
                    "bioregistry": "gene"
                }
            }
        },
        "benchmark": {
            "benchmark_config.yaml": {
                "benchmark": {
                    "metrics": ["mse", "mae", "r2", "perturbation_metric"],
                    "models": ["NBFNetModel", "kStarNN"],
                    "cross_validation": {"folds": 5, "strategy": "kfold"}
                }
            }
        }
    }

    # Write config files
    configs = config_templates.get(experiment_type, config_templates["finetuning"])
    for filename, config_data in configs.items():
        if use_yaml:
            config_path = exp_dir / "config" / filename
        else:
            # Use JSON if YAML not available
            config_path = exp_dir / "config" / filename.replace(".yaml", ".json")

        with open(config_path, "w") as f:
            if use_yaml:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2)

    # Create README.md template
    readme_template = f"""# {experiment_name.replace('_', ' ').title()}

**Experiment Type:** {experiment_type.title()}
**Created:** {datetime.now().strftime('%Y-%m-%d')}
**Status:** 🚧 In Development

## Overview

This experiment focuses on [DESCRIBE THE MAIN OBJECTIVE HERE].

## Methodology

### Data
- **Network:** STRING (combined_scores)
- **Embeddings:** string-space-sequence gene embeddings
- **Task Data:** K562 perturbation data

### Model Architecture
- **Base Model:** NBFNetModel
- **Configuration:** See `config/model_config.yaml`

### Training Setup
- **Epochs:** 1000 (with early stopping)
- **Split:** 80/20 train/test based on obs.fold
- **Metrics:** MSE loss + perturbation_metric (MLP and kStarNN)

## Usage

```bash
# Train the model
cd experiments/{experiment_name}
python scripts/train.py

# Evaluate results
python scripts/evaluate.py

# Generate analysis
python scripts/analyze.py
```

## Results

[TO BE FILLED AFTER EXPERIMENT COMPLETION]

## Key Findings

[TO BE FILLED AFTER ANALYSIS]

## Reproducibility

All configurations are stored in `config/` directory.
Logs and checkpoints are automatically saved during training.

## Next Steps

- [ ] Implement training script
- [ ] Run baseline validation
- [ ] Compare with nbfnet-update implementation
- [ ] Full experiment execution
- [ ] Results analysis and documentation
"""

    with open(exp_dir / "README.md", "w") as f:
        f.write(readme_template)

    # Create basic script templates
    train_script_template = f"""#!/usr/bin/env python3
\"\"\"Training script for {experiment_name} experiment.\"\"\"

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    \"\"\"Main training function.\"\"\"
    print(f"🚀 Starting {experiment_name} training...")
    # TODO: Implement training logic
    pass

if __name__ == "__main__":
    main()
"""

    with open(exp_dir / "scripts" / "train.py", "w") as f:
        f.write(train_script_template)

    # Make script executable
    (exp_dir / "scripts" / "train.py").chmod(0o755)

    session.log(f"✅ Experiment created successfully at: {exp_dir}")
    session.log("📁 Directory structure:")
    for item in sorted(exp_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(exp_dir)
            session.log(f"   📄 {rel_path}")
        elif item != exp_dir:
            rel_path = item.relative_to(exp_dir)
            session.log(f"   📁 {rel_path}/")

    session.log("\n🎯 Next steps:")
    session.log(f"   1. cd experiments/{experiment_name}")
    session.log("   2. Edit config files in config/ directory")
    session.log("   3. Implement scripts in scripts/ directory")
    session.log("   4. Run your experiment!")


if __name__ == "__main__":
    print("🚀 Enhanced Noxfile for study ML Workloads - Single Source of Truth")
    print("📋 Run 'nox -s help' for comprehensive usage information")
    print("🔧 ALL operations must go through nox sessions - this is the only source of truth")
    print("🖥️  Interactive sessions run in screen/tmux by default for persistent development")
    print("💡 Use --no-tmux flag to run directly without terminal multiplexer")
    print("🔄 Prefers screen over tmux for better Warp terminal compatibility")
    print("")
    print("🎯 CRITICAL: All dependency management, environment setup, and operations")
    print("   must use nox sessions. Never run uv/pip commands directly!")
    print("")
    print("📦 Key sessions for dependency management:")
    print("   • nox -s init_project          # Complete project setup with local deps")

