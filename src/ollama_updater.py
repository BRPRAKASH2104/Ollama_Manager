#!/usr/bin/env python3
"""
Cross-Platform Ollama Model Updater (Working Version)
====================================================

A robust cross-platform Python script to update all installed Ollama models.
Fixed all syntax errors and compatible with Python 3.8+

Author: AI Test Case Generator Team
Version: 3.0 (Working)
"""

import asyncio
import subprocess
import sys
import platform
import time
import json
import argparse
import logging
import signal
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import re
from datetime import datetime, timezone
from functools import wraps

# Simple fallback for StrEnum
class StrEnum(str, Enum):
    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

# Type aliases
ModelName = str
CommandResult = Tuple[bool, str, str]
UpdateStatus = str

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

@dataclass(frozen=True)
class ModelInfo:
    """Container for model information"""
    name: str
    tag: str
    model_id: str
    size: str
    modified: str
    
    @property
    def full_name(self) -> ModelName:
        return f"{self.name}:{self.tag}" if self.tag != "latest" else self.name
    
    @property
    def size_bytes(self) -> int:
        size_str = self.size.upper()
        multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
        
        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                try:
                    return int(float(size_str[:-len(unit)]) * multiplier)
                except ValueError:
                    return 0
        return 0

@dataclass
class UpdateResult:
    """Container for update results"""
    model: ModelName
    status: UpdateStatus
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    error_message: str = ""
    retry_count: int = 0
    
    @property
    def duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success(self) -> bool:
        return self.status == "success"
    
    def mark_completed(self, success: bool, error_message: str = ""):
        self.status = "success" if success else "failed"
        self.end_time = datetime.now(timezone.utc)
        self.error_message = error_message
        return self

class EnhancedLogger:
    """Enhanced logging system"""
    
    def __init__(self, verbose: bool = True, log_file: Optional[Path] = None):
        self.verbose = verbose
        self.log_entries: List[Dict] = []
        self.start_time = datetime.now(timezone.utc)
        
        self.logger = logging.getLogger("OllamaUpdater")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        timestamp = datetime.now(timezone.utc)
        
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'elapsed': (timestamp - self.start_time).total_seconds()
        }
        
        self.log_entries.append(log_entry)
        
        log_method = getattr(self.logger, level.lower())
        log_method(message)
    
    def export_logs(self, output_path: Path) -> None:
        serializable_logs = []
        for entry in self.log_entries:
            serializable_entry = entry.copy()
            serializable_entry['timestamp'] = entry['timestamp'].isoformat()
            serializable_logs.append(serializable_entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_logs, f, indent=2)

class OllamaUpdater:
    """Main updater class"""
    
    def __init__(self, verbose: bool = True, dry_run: bool = False, max_concurrent: int = 3, log_file: Optional[Path] = None):
        self.verbose = verbose
        self.dry_run = dry_run
        self.max_concurrent = max_concurrent
        self.system = platform.system()
        self.logger = EnhancedLogger(verbose, log_file)
        self.cancelled = False
        
        self.shell_configs = self._get_shell_config()
        self._setup_signal_handlers()
        
        self.logger.log(f"🚀 Ollama Model Updater v3.0 (Python {sys.version.split()[0]})")
        self.logger.log(f"💻 Platform: {self.system} {platform.release()}")
        
        if self.dry_run:
            self.logger.log("🧪 DRY RUN MODE: No actual updates will be performed")
    
    def _get_shell_config(self) -> Dict:
        base_config = {
            'shell': True,
            'text': True,
            'encoding': 'utf-8'
        }
        
        if self.system == 'Windows':
            try:
                base_config['creationflags'] = subprocess.CREATE_NO_WINDOW
            except AttributeError:
                self.logger.log("⚠️ CREATE_NO_WINDOW not available", LogLevel.WARNING)
        
        return base_config
    
    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            self.logger.log(f"⏹️ Received signal {signum}, shutting down...")
            self.cancelled = True
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (AttributeError, OSError):
            pass
    
    async def run_command_async(self, command: str, capture_output: bool = True, timeout: int = 300) -> CommandResult:
        try:
            if capture_output:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                success = process.returncode == 0
                return success, stdout.decode('utf-8').strip(), stderr.decode('utf-8').strip()
            else:
                process = await asyncio.create_subprocess_shell(command)
                returncode = await asyncio.wait_for(process.wait(), timeout=timeout)
                return returncode == 0, "", ""
                
        except asyncio.TimeoutError:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Error: {e}"
    
    async def check_ollama_available(self) -> bool:
        self.logger.log("🔍 Checking Ollama availability...")
        
        success, stdout, stderr = await self.run_command_async("ollama --version")
        
        if success:
            version_match = re.search(r'ollama version is (\S+)', stdout)
            version = version_match.group(1) if version_match else "unknown"
            self.logger.log(f"✅ Ollama found (version: {version})")
            return True
        else:
            self.logger.log(f"❌ Ollama not found: {stderr}", LogLevel.ERROR)
            return False
    
    def parse_model_list(self, output: str) -> List[ModelInfo]:
        models: List[ModelInfo] = []
        lines = output.strip().split('\n')
        
        if len(lines) < 2:
            return models
        
        for line in lines[1:]:
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                name_tag = parts[0]
                if ':' in name_tag:
                    name, tag = name_tag.rsplit(':', 1)
                else:
                    name, tag = name_tag, "latest"
                
                model = ModelInfo(
                    name=name,
                    tag=tag,
                    model_id=parts[1],
                    size=parts[2],
                    modified=' '.join(parts[3:])
                )
                models.append(model)
                
            except (ValueError, IndexError):
                continue
        
        return models
    
    async def get_installed_models(self) -> List[ModelInfo]:
        self.logger.log("📋 Fetching installed models...")
        
        success, stdout, stderr = await self.run_command_async("ollama list")
        
        if not success:
            self.logger.log(f"❌ Failed to get model list: {stderr}", LogLevel.ERROR)
            return []
        
        models = self.parse_model_list(stdout)
        
        if models:
            self.logger.log(f"✅ Found {len(models)} installed models")
            if self.verbose:
                for model in models:
                    self.logger.log(f"   📦 {model.full_name} ({model.size})")
        else:
            self.logger.log("📭 No models found")
        
        return models
    
    async def update_single_model(self, model: ModelInfo, index: int, total: int, semaphore: asyncio.Semaphore) -> UpdateResult:
        async with semaphore:
            if self.cancelled:
                return UpdateResult(model=model.full_name, status="cancelled")
            
            result = UpdateResult(model=model.full_name, status="running")
            
            self.logger.log(f"\n[{index}/{total}] 🔄 Updating: {model.full_name}")
            
            if self.dry_run:
                self.logger.log(f"🧪 DRY RUN: Would update {model.full_name}")
                await asyncio.sleep(1)
                return result.mark_completed(True)
            
            try:
                success, _, stderr = await self.run_command_async(
                    f'ollama pull {model.full_name}', 
                    capture_output=False,
                    timeout=1800
                )
                
                if success:
                    self.logger.log(f"✅ Successfully updated: {model.full_name} ({result.duration:.1f}s)")
                    return result.mark_completed(True)
                else:
                    error_msg = stderr or "Unknown error"
                    self.logger.log(f"❌ Failed to update: {model.full_name} - {error_msg}", LogLevel.ERROR)
                    return result.mark_completed(False, error_msg)
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.log(f"❌ Exception updating {model.full_name}: {error_msg}", LogLevel.ERROR)
                return result.mark_completed(False, error_msg)
    
    async def select_models_interactively(self, models: List[ModelInfo]) -> List[ModelInfo]:
        if not models:
            return []
        
        print(f"\n📋 Available models ({len(models)} total):")
        
        # Group models by size
        large_models = [m for m in models if m.size_bytes > 10 * 1024**3]
        medium_models = [m for m in models if 1024**3 <= m.size_bytes <= 10 * 1024**3]
        small_models = [m for m in models if m.size_bytes < 1024**3]
        
        size_groups = {
            'Large (>10GB)': large_models,
            'Medium (1-10GB)': medium_models,
            'Small (<1GB)': small_models
        }
        
        model_index = 1
        index_to_model = {}
        
        for group_name, group_models in size_groups.items():
            if group_models:
                print(f"\n   {group_name}:")
                for model in sorted(group_models, key=lambda m: m.name):
                    print(f"   {model_index:2d}. {model.full_name} ({model.size})")
                    index_to_model[model_index] = model
                    model_index += 1
        
        print(f"\n❓ Select models to update:")
        print(f"   • Enter numbers (e.g., 1,3,5) for specific models")
        print(f"   • Enter 'all' or 'a' for all models")
        print(f"   • Enter 'large', 'medium', 'small' for size groups")
        print(f"   • Enter 'q' to quit")
        
        try:
            selection = input(f"\nSelection [all]: ").strip().lower()
            
            if selection in ['q', 'quit']:
                return []
            
            if selection in ['', 'all', 'a']:
                return models
            
            if selection == 'large':
                return large_models
            elif selection == 'medium':
                return medium_models
            elif selection == 'small':
                return small_models
            else:
                # Parse specific model numbers
                selected_models = []
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
            print(f"\n⏹️ Selection cancelled")
            return []
    
    def generate_detailed_report(self, results: List[UpdateResult]) -> Dict:
        total_duration = time.time() - self.logger.start_time.timestamp()
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
                'average_update_time': sum(r.duration for r in successful) / len(successful) if successful else 0
            },
            'successful_updates': [{'model': r.model, 'duration': r.duration} for r in successful],
            'failed_updates': [{'model': r.model, 'error': r.error_message} for r in failed],
            'cancelled_updates': [r.model for r in cancelled],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'platform': {
                'system': self.system,
                'python_version': sys.version
            }
        }
    
    def export_report(self, report: Dict, output_path: Path) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            self.logger.log(f"📄 Report exported to: {output_path}")
        except Exception as e:
            self.logger.log(f"❌ Failed to export report: {e}", LogLevel.ERROR)
    
    async def update_models_async(self, models: List[ModelInfo]) -> List[UpdateResult]:
        if not models:
            self.logger.log("📭 No models to update")
            return []
        
        self.logger.log(f"🚀 Starting update of {len(models)} models...")
        self.logger.log(f"⚙️ Max concurrent updates: {self.max_concurrent}")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        tasks = [
            self.update_single_model(model, i, len(models), semaphore)
            for i, model in enumerate(models, 1)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.log(f"❌ Task {i+1} failed: {result}", LogLevel.ERROR)
                    final_results.append(UpdateResult(
                        model=models[i].full_name,
                        status="failed",
                        error_message=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        except KeyboardInterrupt:
            self.logger.log("⏹️ Update process interrupted", LogLevel.WARNING)
            return []
    
    async def run_update_process(self, strategy: UpdateStrategy = UpdateStrategy.SELECTIVE, export_report: bool = True) -> None:
        try:
            if not await self.check_ollama_available():
                return
            
            models = await self.get_installed_models()
            if not models:
                return
            
            # Select models based on strategy
            if strategy == UpdateStrategy.ALL:
                selected_models = models
            elif strategy == UpdateStrategy.SELECTIVE:
                selected_models = await self.select_models_interactively(models)
            else:
                selected_models = models
            
            if not selected_models:
                self.logger.log("⏹️ No models selected for update")
                return
            
            # Confirm update
            if not self.dry_run:
                total_size = sum(model.size_bytes for model in selected_models)
                size_str = self._format_bytes(total_size)
                
                print(f"\n❓ About to update {len(selected_models)} models (total size: {size_str})")
                try:
                    confirm = input("   Proceed? [Y/n]: ").strip().lower()
                    if confirm in ['n', 'no']:
                        self.logger.log("⏹️ Update cancelled by user")
                        return
                except (KeyboardInterrupt, EOFError):
                    self.logger.log("\n⏹️ Update cancelled by user")
                    return
            
            # Perform updates
            results = await self.update_models_async(selected_models)
            
            # Generate report
            if results:
                report = self.generate_detailed_report(results)
                self._display_summary(report)
                
                if export_report:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_path = Path(f"ollama_update_report_{timestamp}.json")
                    self.export_report(report, report_path)
            
        except Exception as e:
            self.logger.log(f"💥 Unexpected error: {e}", LogLevel.CRITICAL)
            raise
    
    def _format_bytes(self, bytes_value: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f} PB"
    
    def _display_summary(self, report: Dict) -> None:
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
        
        if report['failed_updates']:
            print("\n❌ Failed Updates:")
            for failed in report['failed_updates']:
                print(f"   • {failed['model']}: {failed['error']}")
        
        if summary['failed'] == 0 and summary['cancelled'] == 0:
            print("🎉 All selected models updated successfully!")

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-Platform Ollama Model Updater",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true', help='Update all models')
    parser.add_argument('--strategy', choices=['all', 'selective'], default='selective')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent updates')
    parser.add_argument('--dry-run', action='store_true', help='Preview without updating')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--log-file', type=Path, help='Log to file')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    parser.add_argument('--check-system', action='store_true', help='Check system compatibility')
    parser.add_argument('--list-models', action='store_true', help='List models and exit')
    parser.add_argument('--version', action='version', version='Ollama Model Updater v3.0')
    
    return parser

def check_python_version() -> bool:
    required = (3, 8, 0)
    current = sys.version_info[:3]
    
    if current < required:
        print(f"❌ Python {'.'.join(map(str, required))}+ required")
        print(f"   Current: Python {'.'.join(map(str, current))}")
        return False
    
    return True

async def main() -> int:
    if not check_python_version():
        return 1
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    try:
        if args.check_system:
            print("🔍 System Compatibility Check")
            print("=" * 40)
            print(f"✅ Python: {sys.version}")
            print(f"✅ Platform: {platform.platform()}")
            
            required_modules = ['asyncio', 'subprocess', 'pathlib', 'dataclasses']
            for module in required_modules:
                try:
                    __import__(module)
                    print(f"✅ Module {module}: Available")
                except ImportError:
                    print(f"❌ Module {module}: Missing")
                    return 1
            
            updater = OllamaUpdater(verbose=verbose, log_file=args.log_file)
            if await updater.check_ollama_available():
                print("✅ Ollama: Available")
                return 0
            else:
                print("❌ Ollama: Not available")
                return 1
        
        updater = OllamaUpdater(
            verbose=verbose,
            dry_run=args.dry_run,
            max_concurrent=args.max_concurrent,
            log_file=args.log_file
        )
        
        if args.list_models:
            models = await updater.get_installed_models()
            if not models:
                print("📭 No models installed")
            else:
                print(f"📋 Installed Models ({len(models)}):")
                for model in models:
                    print(f"   📦 {model.full_name} ({model.size})")
            return 0
        
        strategy = UpdateStrategy.ALL if args.all else UpdateStrategy(args.strategy)
        
        await updater.run_update_process(
            strategy=strategy,
            export_report=not args.no_report
        )
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

def sync_main() -> int:
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n⏹️ Operation interrupted")
        return 130
    except Exception as e:
        print(f"💥 Critical error: {e}")
        return 1

if __name__ == "__main__":
    if platform.system() != 'Windows':
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except (ImportError, AttributeError):
            pass
    
    exit_code = sync_main()
    sys.exit(exit_code)