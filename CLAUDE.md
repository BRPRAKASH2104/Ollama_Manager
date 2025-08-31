# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Cross-Platform Ollama Model Updater - a Python script that efficiently updates all installed Ollama models across Windows, macOS, and Linux platforms. The project uses only Python standard library modules (no external dependencies) and provides async operations, interactive model selection, progress tracking, and comprehensive reporting.

## Key Architecture

**Core Components:**
- `OllamaUpdater` - Main orchestrator class handling async operations, concurrency control, and platform-specific configurations
- `ModelInfo` - Immutable dataclass for model information with size calculations and parsing utilities
- `UpdateResult` - Progress tracking container with timing and status management
- `EnhancedLogger` - Structured logging system with JSON export and console/file output
- CLI interface with interactive model selection and argument parsing

**Design Patterns:**
- Async/Await with semaphore-based concurrency control (default: 3 concurrent updates)
- Strategy pattern for update modes (all/selective)
- Observer pattern for progress tracking
- Graceful signal handling for interruption

## Development Commands

**Run the updater:**
```bash
python3 src/ollama_updater.py
```

**Test system compatibility:**
```bash
python3 src/ollama_updater.py --check-system
```

**List installed models:**
```bash
python3 src/ollama_updater.py --list-models
```

**Dry run (preview without updating):**
```bash
python3 src/ollama_updater.py --dry-run
```

**Update all models:**
```bash
python3 src/ollama_updater.py --all
```

**Performance tuning:**
```bash
python3 src/ollama_updater.py --max-concurrent 5 --log-file update.log
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

- Platform-specific shell configuration (handles Windows CREATE_NO_WINDOW properly)
- Async subprocess execution with timeout handling (30 minutes per model)
- Interactive model selection with size-based grouping (Large >10GB, Medium 1-10GB, Small <1GB)
- JSON report generation with timestamp-based filenames
- Signal handling for graceful shutdown on SIGINT/SIGTERM
- Error isolation prevents cascade failures during concurrent updates