#!/usr/bin/env python3.13
"""
Modern Cross-Platform Ollama Model Updater
==========================================

Optimized for Python 3.13.5+ with latest language features.
No backward compatibility - uses cutting-edge Python features.

Author: AI Test Case Generator Team  
Version: 4.0 (Modern)
Requires: Python 3.13.5+
"""

import asyncio
import json
import logging
import platform
import re
import signal
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Annotated, Any, Deque, Final, Literal, Never, Self, 
    TypeAlias, TypeVar, assert_never
)
import argparse

# Modern type aliases using PEP 613 syntax
ModelName: TypeAlias = str
CommandArgs: TypeAlias = list[str]
CommandResult: TypeAlias = tuple[bool, str, str]
UpdateStatus: TypeAlias = Literal["pending", "running", "success", "failed", "cancelled"]

# Generic type for profiling decorator
F = TypeVar('F', bound=callable)

class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class UpdateStrategy(StrEnum):
    ALL = "all"
    SELECTIVE = "selective"
    FAILED_ONLY = "failed_only"
    RECENT_ONLY = "recent_only"

class OllamaError(Exception):
    """Base exception for Ollama operations."""
    pass

class ModelUpdateError(OllamaError):
    """Specific error for model update failures."""
    def __init__(self, model: str, original_error: Exception):
        self.model = model
        self.original_error = original_error
        super().__init__(f"Failed to update {model}: {original_error}")

@dataclass(frozen=True)  # Remove slots to allow cached_property
class ModelInfo:
    """Immutable container for model information with optimized memory usage."""
    name: str
    tag: str
    model_id: str
    size: str
    modified: str
    
    @property
    def full_name(self) -> ModelName:
        return f"{self.name}:{self.tag}" if self.tag != "latest" else self.name
    
    @cached_property  # Cache expensive calculation
    def size_bytes(self) -> int:
        """Convert size string to bytes using modern string methods."""
        size_str = self.size.upper()
        multipliers: Final = {
            'B': 1, 'KB': 1024, 'MB': 1024**2, 
            'GB': 1024**3, 'TB': 1024**4, 'PB': 1024**5
        }
        
        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                try:
                    # Use removesuffix (Python 3.9+) for efficiency
                    number_str = size_str.removesuffix(unit)
                    return int(float(number_str) * multiplier)
                except ValueError:
                    return 0
        return 0

@dataclass  # Remove slots to allow cached_property
class UpdateResult:
    """Container for update results with optimized memory usage."""
    model: ModelName
    status: UpdateStatus
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    error_message: str = ""
    retry_count: int = 0
    
    @cached_property
    def duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success(self) -> bool:
        return self.status == "success"
    
    def mark_completed(self, success: bool, error_message: str = "") -> Self:
        """Mark update as completed with modern Self return type."""
        self.status = "success" if success else "failed"
        self.end_time = datetime.now(timezone.utc)
        self.error_message = error_message
        return self

def profile_async(func: F) -> F:
    """Async profiling decorator using modern type hints."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            print(f"⏱️  {func.__name__} took {duration:.3f}s")
    return wrapper

class EnhancedLogger:
    """Modern logging system with memory-efficient storage."""
    
    MAX_LOG_ENTRIES: Final = 1000
    
    def __init__(self, verbose: bool = True, log_file: Path | None = None):
        self.verbose = verbose
        # Use deque with maxlen for automatic memory management
        self.log_entries: Deque[dict[str, Any]] = deque(maxlen=self.MAX_LOG_ENTRIES)
        self.start_time = datetime.now(timezone.utc)
        
        # Modern logger setup
        self.logger = logging.getLogger("OllamaUpdater")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Console handler with modern formatting
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log message with structured data storage."""
        timestamp = datetime.now(timezone.utc)
        
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'elapsed': (timestamp - self.start_time).total_seconds()
        }
        
        self.log_entries.append(log_entry)
        
        # Use match statement for log level handling (Python 3.10+)
        match level:
            case LogLevel.DEBUG:
                self.logger.debug(message)
            case LogLevel.INFO:
                self.logger.info(message)
            case LogLevel.WARNING:
                self.logger.warning(message)
            case LogLevel.ERROR:
                self.logger.error(message)
            case LogLevel.CRITICAL:
                self.logger.critical(message)
            case _:
                assert_never(level)  # Exhaustiveness check
    
    def export_logs(self, output_path: Path) -> None:
        """Export logs with optimized JSON serialization."""
        serializable_logs = [
            {**entry, 'timestamp': entry['timestamp'].isoformat()}
            for entry in self.log_entries
        ]
        
        # Use modern Path methods
        output_path.write_text(
            json.dumps(serializable_logs, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

class OllamaUpdater:
    """Modern updater with Python 3.13.5+ optimizations."""
    
    def __init__(
        self, 
        verbose: bool = True, 
        dry_run: bool = False, 
        max_concurrent: Annotated[int, "Range 1-20"] = 3, 
        log_file: Path | None = None
    ):
        self.verbose = verbose
        self.dry_run = dry_run
        self.max_concurrent = max_concurrent
        self.system = platform.system()
        self.logger = EnhancedLogger(verbose, log_file)
        self.cancelled = False
        
        self._setup_signal_handlers()
        
        self.logger.log(f"🚀 Modern Ollama Updater v4.0 (Python {sys.version.split()[0]})")
        self.logger.log(f"💻 Platform: {self.system} {platform.release()}")
        
        if self.dry_run:
            self.logger.log("🧪 DRY RUN MODE: No actual updates will be performed")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers with modern exception handling."""
        def signal_handler(signum: int, frame) -> None:
            self.logger.log(f"⏹️ Received signal {signum}, shutting down...")
            self.cancelled = True
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (AttributeError, OSError):
            pass
    
    @asynccontextmanager
    async def ollama_session(self):
        """Context manager for Ollama operations with proper cleanup."""
        try:
            if not await self.check_ollama_available():
                raise OllamaError("Ollama not available")
            self.logger.log("🔧 Ollama session started")
            yield self
        except Exception as e:
            self.logger.log(f"❌ Session error: {e}", LogLevel.ERROR)
            raise
        finally:
            self.logger.log("🔧 Ollama session ended")
    
    @profile_async
    async def run_command_async(
        self, 
        command: CommandArgs, 
        timeout: int = 300
    ) -> CommandResult:
        """Execute command with modern subprocess and security."""
        try:
            # Use create_subprocess_exec for security (no shell injection)
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            success = process.returncode == 0
            return (
                success, 
                stdout.decode('utf-8').strip(), 
                stderr.decode('utf-8').strip()
            )
                
        except asyncio.TimeoutError:
            if process:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Error: {e}"
    
    async def check_ollama_available(self) -> bool:
        """Check Ollama availability with modern error handling."""
        self.logger.log("🔍 Checking Ollama availability...")
        
        success, stdout, stderr = await self.run_command_async(["ollama", "--version"])
        
        match success:
            case True:
                # Use walrus operator for efficient regex matching
                if version_match := re.search(r'ollama version is (\S+)', stdout):
                    version = version_match.group(1)
                else:
                    version = "unknown"
                self.logger.log(f"✅ Ollama found (version: {version})")
                return True
            case False:
                self.logger.log(f"❌ Ollama not found: {stderr}", LogLevel.ERROR)
                return False
    
    def parse_model_list(self, output: str) -> list[ModelInfo]:
        """Parse model list with modern list comprehensions and error handling."""
        lines = output.strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        models: list[ModelInfo] = []
        
        for line in lines[1:]:  # Skip header
            if not (stripped_line := line.strip()):
                continue
                
            parts = stripped_line.split()
            if len(parts) < 5:
                continue
            
            try:
                name_tag = parts[0]
                # Use partition for efficient string splitting
                if ':' in name_tag:
                    name, _, tag = name_tag.rpartition(':')
                else:
                    name, tag = name_tag, "latest"
                
                model = ModelInfo(
                    name=name,
                    tag=tag,
                    model_id=parts[1],
                    size=f"{parts[2]} {parts[3]}" if len(parts) >= 4 else parts[2],
                    modified=' '.join(parts[4:]) if len(parts) > 4 else ""
                )
                models.append(model)
                
            except (ValueError, IndexError):
                continue
        
        return models
    
    @profile_async
    async def get_installed_models(self) -> list[ModelInfo]:
        """Get installed models with modern async patterns."""
        self.logger.log("📋 Fetching installed models...")
        
        success, stdout, stderr = await self.run_command_async(["ollama", "list"])
        
        if not success:
            self.logger.log(f"❌ Failed to get model list: {stderr}", LogLevel.ERROR)
            return []
        
        models = self.parse_model_list(stdout)
        
        match len(models):
            case 0:
                self.logger.log("📭 No models found")
            case n:
                self.logger.log(f"✅ Found {n} installed models")
                if self.verbose:
                    for model in models:
                        self.logger.log(f"   📦 {model.full_name} ({model.size})")
        
        return models
    
    @profile_async
    async def update_single_model(
        self, 
        model: ModelInfo, 
        index: int, 
        total: int, 
        semaphore: asyncio.Semaphore
    ) -> UpdateResult:
        """Update single model with modern concurrency control."""
        async with semaphore:
            if self.cancelled:
                return UpdateResult(model=model.full_name, status="cancelled")
            
            result = UpdateResult(model=model.full_name, status="running")
            
            self.logger.log(f"\n[{index}/{total}] 🔄 Updating: {model.full_name}")
            
            if self.dry_run:
                self.logger.log(f"🧪 DRY RUN: Would update {model.full_name}")
                await asyncio.sleep(1)  # Simulate work
                return result.mark_completed(True)
            
            try:
                success, _, stderr = await self.run_command_async(
                    ["ollama", "pull", model.full_name],
                    timeout=1800
                )
                
                match success:
                    case True:
                        self.logger.log(
                            f"✅ Successfully updated: {model.full_name} "
                            f"({result.duration:.1f}s)"
                        )
                        return result.mark_completed(True)
                    case False:
                        error_msg = stderr or "Unknown error"
                        self.logger.log(
                            f"❌ Failed to update: {model.full_name} - {error_msg}", 
                            LogLevel.ERROR
                        )
                        return result.mark_completed(False, error_msg)
                        
            except Exception as e:
                error_msg = str(e)
                self.logger.log(
                    f"❌ Exception updating {model.full_name}: {error_msg}", 
                    LogLevel.ERROR
                )
                return result.mark_completed(False, error_msg)
    
    async def select_models_interactively(
        self, 
        models: list[ModelInfo]
    ) -> list[ModelInfo]:
        """Interactive model selection with modern pattern matching."""
        if not models:
            return []
        
        print(f"\n📋 Available models ({len(models)} total):")
        
        # Group models by size with modern comprehensions
        size_threshold_gb = 1024**3
        large_models = [m for m in models if m.size_bytes > 10 * size_threshold_gb]
        medium_models = [
            m for m in models 
            if size_threshold_gb <= m.size_bytes <= 10 * size_threshold_gb
        ]
        small_models = [m for m in models if m.size_bytes < size_threshold_gb]
        
        size_groups = {
            'Large (>10GB)': large_models,
            'Medium (1-10GB)': medium_models,
            'Small (<1GB)': small_models
        }
        
        model_index = 1
        index_to_model: dict[int, ModelInfo] = {}
        
        for group_name, group_models in size_groups.items():
            if group_models:
                print(f"\n   {group_name}:")
                for model in sorted(group_models, key=lambda m: m.name):
                    print(f"   {model_index:2d}. {model.full_name} ({model.size})")
                    index_to_model[model_index] = model
                    model_index += 1
        
        print("\n❓ Select models to update:")
        print("   • Enter numbers (e.g., 1,3,5) for specific models")
        print("   • Enter 'all' or 'a' for all models")
        print("   • Enter 'large', 'medium', 'small' for size groups")
        print("   • Enter 'q' to quit")
        
        try:
            selection = input("\nSelection [all]: ").strip().lower()
            
            # Use match statement for selection handling
            match selection:
                case 'q' | 'quit':
                    return []
                case '' | 'all' | 'a':
                    return models
                case 'large':
                    return large_models
                case 'medium':
                    return medium_models
                case 'small':
                    return small_models
                case _:
                    # Parse specific model numbers
                    selected_models: list[ModelInfo] = []
                    for part in selection.split(','):
                        try:
                            num = int(part.strip())
                            if num in index_to_model:
                                selected_models.append(index_to_model[num])
                            else:
                                print(f"⚠️ Invalid model number: {num}")
                        except ValueError:
                            print(f"⚠️ Invalid input: {part}")
                    
                    return selected_models
            
        except (KeyboardInterrupt, EOFError):
            print("\n⏹️ Selection cancelled")
            return []
    
    @profile_async
    async def update_models_async(self, models: list[ModelInfo]) -> list[UpdateResult]:
        """Update models using modern TaskGroup for structured concurrency."""
        if not models:
            self.logger.log("📭 No models to update")
            return []
        
        self.logger.log(f"🚀 Starting update of {len(models)} models...")
        self.logger.log(f"⚙️ Max concurrent updates: {self.max_concurrent}")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        tasks = []
        interrupted = False
        
        try:
            # Use TaskGroup for structured concurrency (Python 3.11+)
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self.update_single_model(model, i, len(models), semaphore),
                        name=f"update_{model.full_name.replace(':', '_')}"
                    )
                    for i, model in enumerate(models, 1)
                ]
            
            # All tasks completed successfully
            return [task.result() for task in tasks]
            
        except* (ModelUpdateError, Exception) as eg:  # Exception group handling
            self.logger.log("❌ Some model updates encountered issues:", LogLevel.ERROR)
            
            # Check for KeyboardInterrupt in exception group
            for exc in eg.exceptions:
                if isinstance(exc, KeyboardInterrupt):
                    self.logger.log("⏹️ Update process interrupted", LogLevel.WARNING)
                    interrupted = True
                    break
        
        # Handle interruption or collect results outside except* block
        if interrupted:
            return []
        
        # Collect results from completed/failed tasks
        results: list[UpdateResult] = []
        for i, task in enumerate(tasks):
            try:
                results.append(task.result())
            except Exception as e:
                self.logger.log(f"❌ Task {i+1} failed: {e}", LogLevel.ERROR)
                results.append(UpdateResult(
                    model=models[i].full_name,
                    status="failed",
                    error_message=str(e)
                ))
        
        return results
    
    def generate_detailed_report(self, results: list[UpdateResult]) -> dict[str, Any]:
        """Generate report with modern type hints and comprehensions."""
        total_duration = time.time() - self.logger.start_time.timestamp()
        
        # Use modern filtering with comprehensions
        successful = [r for r in results if r.success]
        failed = [r for r in results if r.status == "failed"]
        cancelled = [r for r in results if r.status == "cancelled"]
        
        return {
            'summary': {
                'total_models': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'cancelled': len(cancelled),
                'success_rate': len(successful) / len(results) * 100 if results else 0,
                'total_duration': total_duration,
                'average_update_time': (
                    sum(r.duration for r in successful) / len(successful) 
                    if successful else 0
                )
            },
            'successful_updates': [
                {'model': r.model, 'duration': r.duration} 
                for r in successful
            ],
            'failed_updates': [
                {'model': r.model, 'error': r.error_message} 
                for r in failed
            ],
            'cancelled_updates': [r.model for r in cancelled],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'platform': {
                'system': self.system,
                'python_version': sys.version.split()[0]
            }
        }
    
    def export_report(self, report: dict[str, Any], output_path: Path) -> None:
        """Export report using modern Path methods."""
        try:
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            self.logger.log(f"📄 Report exported to: {output_path}")
        except Exception as e:
            self.logger.log(f"❌ Failed to export report: {e}", LogLevel.ERROR)
    
    @profile_async
    async def run_update_process(
        self, 
        strategy: UpdateStrategy = UpdateStrategy.SELECTIVE,
        export_report: bool = True
    ) -> None:
        """Main update process with modern context management."""
        async with self.ollama_session():
            models = await self.get_installed_models()
            if not models:
                return
            
            # Select models using modern match statement
            match strategy:
                case UpdateStrategy.ALL:
                    selected_models = models
                case UpdateStrategy.SELECTIVE:
                    selected_models = await self.select_models_interactively(models)
                case _:
                    selected_models = models
            
            if not selected_models:
                self.logger.log("⏹️ No models selected for update")
                return
            
            # Confirm update with modern walrus operator
            if not self.dry_run:
                total_size = sum(model.size_bytes for model in selected_models)
                size_str = self._format_bytes(total_size)
                
                print(f"\n❓ About to update {len(selected_models)} models "
                      f"(total size: {size_str})")
                try:
                    if (confirm := input("   Proceed? [Y/n]: ").strip().lower()) in ['n', 'no']:
                        self.logger.log("⏹️ Update cancelled by user")
                        return
                except (KeyboardInterrupt, EOFError):
                    self.logger.log("\n⏹️ Update cancelled by user")
                    return
            
            # Perform updates
            results = await self.update_models_async(selected_models)
            
            # Generate and export report
            if results:
                report = self.generate_detailed_report(results)
                self._display_summary(report)
                
                if export_report:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_path = Path(f"ollama_update_report_{timestamp}.json")
                    self.export_report(report, report_path)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes with modern iteration."""
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        
        for unit in units:
            if bytes_value < 1024:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024
        
        return f"{bytes_value:.1f} EB"
    
    def _display_summary(self, report: dict[str, Any]) -> None:
        """Display summary with modern formatting."""
        summary = report['summary']
        
        print("\n" + "=" * 60)
        print("📊 UPDATE SUMMARY")
        print("=" * 60)
        print(f"📋 Total Models: {summary['total_models']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏹️ Cancelled: {summary['cancelled']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {summary['total_duration']:.1f}s")
        
        if summary['successful'] > 0:
            print(f"⚡ Avg Update Time: {summary['average_update_time']:.1f}s")
        
        print("=" * 60)
        
        # Display failures if any
        if failed_updates := report.get('failed_updates'):
            print("\n❌ Failed Updates:")
            for failed in failed_updates:
                print(f"   • {failed['model']}: {failed['error']}")
        
        if summary['failed'] == 0 and summary['cancelled'] == 0:
            print("🎉 All selected models updated successfully!")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with modern type validation."""
    parser = argparse.ArgumentParser(
        description="Modern Cross-Platform Ollama Model Updater (Python 3.13.5+)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true', help='Update all models')
    parser.add_argument(
        '--strategy', 
        type=UpdateStrategy, 
        choices=list(UpdateStrategy), 
        default=UpdateStrategy.SELECTIVE
    )
    parser.add_argument(
        '--max-concurrent', 
        type=int, 
        default=3,
        choices=range(1, 21),
        metavar="1-20",
        help='Max concurrent updates (1-20)'
    )
    parser.add_argument('--dry-run', action='store_true', help='Preview without updating')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--log-file', type=Path, help='Log to file')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    parser.add_argument('--check-system', action='store_true', help='Check system compatibility')
    parser.add_argument('--list-models', action='store_true', help='List models and exit')
    parser.add_argument('--version', action='version', version='Modern Ollama Updater v4.0')
    
    return parser

def check_python_version() -> bool:
    """Check for Python 3.13.5+ requirement."""
    required = (3, 13, 5)
    current = sys.version_info[:3]
    
    if current < required:
        print(f"❌ Python {'.'.join(map(str, required))}+ required for optimal performance")
        print(f"   Current: Python {'.'.join(map(str, current))}")
        print("   This version uses cutting-edge Python features.")
        return False
    
    return True

async def main() -> int:
    """Modern main function with structured error handling."""
    if not check_python_version():
        return 1
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    try:
        if args.check_system:
            print("🔍 Modern System Compatibility Check")
            print("=" * 50)
            print(f"✅ Python: {sys.version}")
            print(f"✅ Platform: {platform.platform()}")
            
            # Check for modern Python features
            features = [
                ('TaskGroup', 'asyncio.TaskGroup'),
                ('Exception Groups', 'ExceptionGroup'),
                ('StrEnum', 'enum.StrEnum'),
                ('Match Statement', 'match/case'),
                ('Walrus Operator', ':='),
                ('Union Types', 'X | Y'),
            ]
            
            for feature_name, _ in features:
                print(f"✅ {feature_name}: Available")
            
            updater = OllamaUpdater(verbose=verbose, log_file=args.log_file)
            async with updater.ollama_session():
                print("✅ Ollama: Available")
                return 0
        
        updater = OllamaUpdater(
            verbose=verbose,
            dry_run=args.dry_run,
            max_concurrent=args.max_concurrent,
            log_file=args.log_file
        )
        
        if args.list_models:
            async with updater.ollama_session():
                models = await updater.get_installed_models()
                match len(models):
                    case 0:
                        print("📭 No models installed")
                    case n:
                        print(f"📋 Installed Models ({n}):")
                        for model in models:
                            print(f"   📦 {model.full_name} ({model.size})")
                return 0
        
        strategy = UpdateStrategy.ALL if args.all else args.strategy
        
        await updater.run_update_process(
            strategy=strategy,
            export_report=not args.no_report
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

def sync_main() -> Never:
    """Entry point with modern error handling."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Modern signal handling for non-Windows platforms
    if platform.system() != 'Windows':
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except (ImportError, AttributeError):
            pass
    
    sync_main()