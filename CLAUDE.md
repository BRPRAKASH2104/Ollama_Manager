# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Cross-Platform Ollama Model Updater - a modern Python script that efficiently updates all installed Ollama models across Windows, macOS, and Linux platforms. The project is designed as a zero-dependency solution using only Python standard library modules, featuring async operations, interactive model selection, progress tracking, and comprehensive reporting.

## Architecture Overview

**Single-File Design:**
The entire application is contained in `src/ollama_updater.py` (~1000+ lines) using modern Python 3.13.5+ features. This monolithic design choice ensures zero external dependencies while providing enterprise-grade functionality.

**Core Classes:**
- `OllamaUpdater` - Main orchestrator using async context managers and semaphore-based concurrency
- `ModelInfo` - Immutable frozen dataclass with cached properties for memory-efficient model metadata
- `UpdateResult` - State tracking container with datetime management and completion markers
- `EnhancedLogger` - Memory-bounded logging using deque with automatic rotation (max 1000 entries)

**Modern Python Features Used:**
- Type aliases with `TypeAlias` annotation (PEP 613)
- String enums (`StrEnum`) for type safety
- Match statements for exhaustive pattern matching (Python 3.10+)
- `Self` return type annotation for method chaining
- `@asynccontextmanager` for resource management
- `@cached_property` for expensive computations
- `assert_never()` for exhaustiveness checking
- Annotated types for parameter constraints

**Concurrency Model:**
- Asyncio-based with semaphore limiting concurrent updates (default: 3)
- Each model update runs as isolated async task
- Graceful cancellation via signal handlers (SIGINT/SIGTERM)
- Exception isolation prevents cascade failures
- Memory-efficient with automatic resource cleanup

## Development Commands

**Run the updater interactively:**
```bash
python3 src/ollama_updater.py
```

**System verification and testing:**
```bash
# Check Python version and module availability
python3 src/ollama_updater.py --check-system

# Preview updates without executing
python3 src/ollama_updater.py --dry-run

# List and analyze installed models
python3 src/ollama_updater.py --list-models
```

**Update operations:**
```bash
# Update all models (non-interactive)
python3 src/ollama_updater.py --all

# Update with specific strategy
python3 src/ollama_updater.py --strategy selective

# Performance tuning with logging
python3 src/ollama_updater.py --max-concurrent 5 --log-file update.log --verbose
```

**Development and debugging:**
```bash
# Verbose output with detailed logs
python3 src/ollama_updater.py --verbose --log-file debug.log

# Quiet mode for scripting
python3 src/ollama_updater.py --all --quiet

# Version and help information
python3 src/ollama_updater.py --version
python3 src/ollama_updater.py --help
```

## Requirements

- Python 3.8+ (3.13.5+ recommended)
- Ollama installed and accessible via command line
- No external Python dependencies (uses only standard library)

## File Structure

- `src/ollama_updater.py` - Main updater script with all functionality
- `src/requirements.txt` - Documentation of built-in modules (no pip install needed)
- `README.md` - Comprehensive documentation
- `ollama_updater_readme_txt.txt` - Duplicate documentation in text format

## Testing and Validation

Always test changes with:
1. `--check-system` to verify environment compatibility
2. `--dry-run` to preview operations without executing
3. `--list-models` to verify model parsing
4. Test on different platforms (Windows/macOS/Linux) if modifying platform-specific code

## Key Implementation Details

**Modern Python Architecture (v4.0):**
- Uses Python 3.13.5+ exclusive features with no backward compatibility
- Type-safe design with comprehensive type aliases and enums
- Memory-optimized with cached properties and bounded collections
- Security-focused with parameterized subprocess calls (no shell injection)

**Core Patterns:**
- `@asynccontextmanager` for resource lifecycle management
- Semaphore-based concurrency control with configurable limits
- Exception hierarchy with custom `OllamaError` and `ModelUpdateError` classes
- Signal handling for graceful shutdown on SIGINT/SIGTERM
- Profiling decorator for performance monitoring

**Data Flow:**
1. **Discovery:** Parse `ollama list` output into `ModelInfo` objects with size calculation
2. **Selection:** Interactive UI with size-based grouping (Large >10GB, Medium 1-10GB, Small <1GB)
3. **Execution:** Async task creation with semaphore-controlled concurrency
4. **Tracking:** Real-time status updates via `UpdateResult` containers
5. **Reporting:** JSON export with comprehensive metrics and structured logging