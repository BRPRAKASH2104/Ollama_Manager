================================================================================
                    CROSS-PLATFORM OLLAMA MODEL UPDATER
================================================================================

Version: 3.0 (Working)
Author: AI Test Case Generator Team
License: MIT
Last Updated: December 19, 2024

================================================================================
                                OVERVIEW
================================================================================

The Cross-Platform Ollama Model Updater is a robust Python script designed to
efficiently update all installed Ollama models across Windows, macOS, and Linux
platforms. It provides advanced features including async operations, interactive
model selection, progress tracking, and comprehensive reporting.

KEY FEATURES:
- Cross-platform compatibility (Windows, macOS, Linux)
- Async operations for concurrent model updates
- Interactive model selection with size-based grouping
- Real-time progress tracking and status updates
- Comprehensive JSON reporting and structured logging
- Dry-run mode for safe testing
- Graceful signal handling and error recovery
- Built using only Python standard library (no external dependencies)

================================================================================
                            SYSTEM REQUIREMENTS
================================================================================

MINIMUM REQUIREMENTS:
- Python 3.8.0 or higher (Python 3.13.5+ recommended for best performance)
- Ollama installed and accessible via command line
- Internet connection for downloading model updates
- Minimum 2GB RAM (4GB+ recommended for large models)
- Adequate disk space for model downloads (10GB+ recommended)

SUPPORTED PLATFORMS:
- Windows 10/11 (x64)
- macOS 10.15+ (Intel and Apple Silicon)
- Linux (Ubuntu 18.04+, CentOS 7+, Debian 10+, or equivalent)

PYTHON MODULES REQUIRED:
All dependencies are part of Python's standard library:
- asyncio (async operations)
- subprocess (command execution)
- json (data handling)
- argparse (CLI interface)
- pathlib (file operations)
- dataclasses (data containers)
- typing (type hints)
- logging (structured logging)
- signal (graceful shutdown)
- re (pattern matching)
- datetime (timestamps)
- functools (utilities)

================================================================================
                               INSTALLATION
================================================================================

STEP 1: VERIFY PYTHON VERSION
Check that you have Python 3.8 or higher installed:

    python3 --version

Expected output: Python 3.8.0 or higher

STEP 2: VERIFY PYTHON MODULES
All required modules should be available by default:

    python3 -c "
    import asyncio, subprocess, sys, platform, time, json, argparse
    import logging, signal, typing, dataclasses, pathlib, enum, re
    import datetime, functools
    print('✅ All required modules available')
    "

STEP 3: VERIFY OLLAMA INSTALLATION
Ensure Ollama is installed and accessible:

    ollama --version
    ollama list

If Ollama is not installed, visit: https://ollama.com/download

STEP 4: DOWNLOAD THE SCRIPT
Download ollama_updater_working.py to your desired location.

STEP 5: TEST INSTALLATION
Verify everything is working correctly:

    python3 ollama_updater_working.py --check-system

Expected output should show all green checkmarks (✅).

REQUIREMENTS.TXT:
No external dependencies are required. The included requirements.txt file
documents the built-in modules and optional development tools.

================================================================================
                                 USAGE
================================================================================

BASIC COMMANDS:

Interactive Mode (Recommended for first use):
    python3 ollama_updater_working.py

Update All Models:
    python3 ollama_updater_working.py --all

Dry Run (Preview without updating):
    python3 ollama_updater_working.py --dry-run

List Installed Models:
    python3 ollama_updater_working.py --list-models

Check System Compatibility:
    python3 ollama_updater_working.py --check-system

ADVANCED OPTIONS:

Concurrency Control:
    python3 ollama_updater_working.py --max-concurrent 5

Logging:
    python3 ollama_updater_working.py --log-file update.log

Quiet Mode:
    python3 ollama_updater_working.py --quiet

Verbose Mode:
    python3 ollama_updater_working.py --verbose

Help:
    python3 ollama_updater_working.py --help

================================================================================
                            COMMAND LINE OPTIONS
================================================================================

MAIN OPERATIONS:
--all                   Update all models without prompting
--strategy STRATEGY     Update strategy (all|selective) [default: selective]
--dry-run              Show what would be updated without actually updating
--list-models          List installed models and exit
--check-system         Check system compatibility and exit

PERFORMANCE:
--max-concurrent N     Maximum concurrent updates [default: 3]

OUTPUT CONTROL:
--verbose, -v          Enable detailed output [default: enabled]
--quiet, -q           Minimal output (overrides --verbose)
--log-file PATH       Write detailed logs to file
--no-report           Skip generating update report

INFORMATION:
--version             Show version information
--help               Show help message and exit

================================================================================
                               ARCHITECTURE
================================================================================

CORE COMPONENTS:

1. OllamaUpdater (Main Class)
   - Orchestrates the entire update process
   - Manages async operations and concurrency control
   - Handles platform-specific configurations
   - Provides error handling and recovery

2. ModelInfo (Data Container)
   - Immutable container for model information
   - Parses model names, tags, sizes, and metadata
   - Provides utility methods for size calculations

3. UpdateResult (Progress Tracking)
   - Tracks update status and timing
   - Records success/failure states
   - Calculates performance metrics

4. EnhancedLogger (Logging System)
   - Structured logging with multiple levels
   - JSON export capabilities
   - Console and file output support

5. Command Line Interface
   - Argument parsing and validation
   - Interactive model selection
   - User input handling

DESIGN PATTERNS:

- Async/Await Pattern: Enables concurrent model updates
- Observer Pattern: Progress tracking and status updates
- Strategy Pattern: Different update strategies (all, selective)
- Factory Pattern: Component creation and configuration
- Command Pattern: CLI operations and execution

CONCURRENCY MODEL:

The script uses asyncio with semaphore-based concurrency control:
- Semaphore limits simultaneous updates (default: 3)
- Each model update runs as an async task
- Graceful cancellation on user interruption
- Exception isolation prevents cascade failures

================================================================================
                              PROGRAM FLOW
================================================================================

INITIALIZATION PHASE:
1. Parse command line arguments
2. Verify Python version compatibility
3. Initialize OllamaUpdater with configuration
4. Setup signal handlers for graceful shutdown
5. Configure logging system

DISCOVERY PHASE:
6. Check Ollama availability and version
7. Execute "ollama list" command
8. Parse model information from output
9. Group models by size categories
10. Display available models to user

SELECTION PHASE:
11. Present interactive model selection interface
12. Parse user input (numbers, ranges, keywords)
13. Validate selected models
14. Calculate total download size
15. Request user confirmation

UPDATE PHASE:
16. Create async semaphore for concurrency control
17. Generate update tasks for selected models
18. Execute concurrent "ollama pull" commands
19. Track progress and status for each model
20. Handle errors and retry logic

REPORTING PHASE:
21. Collect update results and metrics
22. Generate comprehensive status report
23. Display summary to user
24. Export JSON report and logs (optional)
25. Cleanup and graceful shutdown

ERROR HANDLING:
- Network timeout handling
- Model not found errors
- Disk space validation
- Permission error recovery
- Graceful interruption support

================================================================================
                            INTERACTIVE FEATURES
================================================================================

MODEL SELECTION INTERFACE:

When running in interactive mode, the script presents models grouped by size:

    📋 Available models (8 total):

       Large (>10GB):
        1. llama3.1:70b (40.0GB)
        2. codellama:34b (19.0GB)

       Medium (1-10GB):
        3. llama3.1:8b (4.7GB)
        4. mistral:7b (4.1GB)
        5. phi3:medium (7.9GB)

       Small (<1GB):
        6. gemma:2b (1.4GB)

    ❓ Select models to update:
       • Enter numbers (e.g., 1,3,5) for specific models
       • Enter 'all' or 'a' for all models
       • Enter 'large', 'medium', 'small' for size groups
       • Enter 'q' to quit

    Selection [all]:

INPUT OPTIONS:
- Specific numbers: 1,3,5
- Ranges: 1-5 (not implemented in current version)
- Size groups: large, medium, small
- All models: all, a, or just press Enter
- Quit: q, quit

================================================================================
                              OUTPUT EXAMPLES
================================================================================

SYSTEM CHECK OUTPUT:
    🔍 System Compatibility Check
    ============================
    ✅ Python: 3.11.5
    ✅ Platform: Darwin-23.1.0-arm64-arm-64bit
    ✅ Module asyncio: Available
    ✅ Module subprocess: Available
    ✅ Module pathlib: Available
    ✅ Module dataclasses: Available
    ✅ Ollama: Available

NORMAL UPDATE OUTPUT:
    🚀 Ollama Model Updater v3.0 (Python 3.11.5)
    💻 Platform: Darwin 23.1.0
    🔍 Checking Ollama availability...
    ✅ Ollama found (version: 0.1.17)
    📋 Fetching installed models...
    ✅ Found 3 installed models

    [1/3] 🔄 Updating: llama3.1:8b
    ✅ Successfully updated: llama3.1:8b (45.2s)

    [2/3] 🔄 Updating: mistral:7b
    ✅ Successfully updated: mistral:7b (38.7s)

    ============================================================
    📊 UPDATE SUMMARY
    ============================================================
    📋 Total Models: 3
    ✅ Successful: 3
    ❌ Failed: 0
    ⏹️ Cancelled: 0
    📈 Success Rate: 100.0%
    ⏱️ Total Duration: 126.0s
    ⚡ Avg Update Time: 42.0s
    ============================================================
    🎉 All selected models updated successfully!

DRY RUN OUTPUT:
    🧪 DRY RUN MODE: No actual updates will be performed
    📋 Available models (3 total):
    
    🧪 DRY RUN: Would update llama3.1:8b
    🧪 DRY RUN: Would update mistral:7b
    
    📊 DRY RUN SUMMARY: 2 models would be updated

================================================================================
                            GENERATED REPORTS
================================================================================

JSON REPORT STRUCTURE:
The script generates detailed JSON reports with the following structure:

{
  "summary": {
    "total_models": 3,
    "successful": 3,
    "failed": 0,
    "cancelled": 0,
    "success_rate": 100.0,
    "total_duration": 126.0,
    "average_update_time": 42.0
  },
  "successful_updates": [
    {
      "model": "llama3.1:8b",
      "duration": 45.2
    }
  ],
  "failed_updates": [],
  "cancelled_updates": [],
  "timestamp": "2024-12-19T14:30:22.123456+00:00",
  "platform": {
    "system": "Darwin",
    "python_version": "3.11.5"
  }
}

LOG FILE STRUCTURE:
Structured logs are exported in JSON format:

[
  {
    "timestamp": "2024-12-19T14:30:22.123456+00:00",
    "level": "info",
    "message": "🚀 Ollama Model Updater v3.0",
    "elapsed": 0.0
  }
]

FILE NAMING:
- Reports: ollama_update_report_YYYYMMDD_HHMMSS.json
- Logs: ollama_update_logs_YYYYMMDD_HHMMSS.json

================================================================================
                            TROUBLESHOOTING
================================================================================

COMMON ISSUES AND SOLUTIONS:

1. PYTHON VERSION ERROR:
   Problem: "Python 3.8.0+ required"
   Solution: Upgrade Python to 3.8 or higher
   - macOS: brew install python@3.11
   - Ubuntu: sudo apt install python3.11
   - Windows: Download from python.org

2. OLLAMA NOT FOUND:
   Problem: "Ollama not found or not accessible"
   Solution: Install and configure Ollama
   - Download from: https://ollama.com/download
   - Verify installation: ollama --version
   - Restart terminal after installation

3. PERMISSION ERRORS:
   Problem: "Permission denied" during update
   Solution: Check user permissions
   - Ensure user can execute ollama commands
   - Check disk space availability
   - Run with appropriate permissions

4. NETWORK TIMEOUT:
   Problem: "Command timed out after 1800 seconds"
   Solution: Increase timeout or reduce concurrency
   - Use: --max-concurrent 1
   - Check network connection stability
   - Try updating smaller models first

5. MODEL NOT FOUND:
   Problem: "Failed to update: model not found"
   Solution: Verify model availability
   - Check model name spelling
   - Verify model exists on Ollama hub
   - Try manual pull: ollama pull model-name

6. MEMORY ERRORS:
   Problem: Out of memory during large model updates
   Solution: Reduce concurrent updates
   - Use: --max-concurrent 1
   - Close other applications
   - Update models individually

7. DISK SPACE ERRORS:
   Problem: "No space left on device"
   Solution: Free up disk space
   - Remove old/unused models: ollama rm model-name
   - Clean up temporary files
   - Move models to different location

DEBUG MODE:
Enable detailed logging for troubleshooting:
    python3 ollama_updater_working.py --verbose --log-file debug.log

For persistent issues, check the generated log files for detailed error
information and stack traces.

================================================================================
                           PERFORMANCE TUNING
================================================================================

CONCURRENCY SETTINGS:

Default Configuration:
- Max concurrent updates: 3
- Timeout per model: 1800 seconds (30 minutes)

Fast Network Optimization:
    python3 ollama_updater_working.py --max-concurrent 8

Slow Network Optimization:
    python3 ollama_updater_working.py --max-concurrent 1

RESOURCE USAGE:

Memory Usage:
- Base script: ~20-50MB
- Per concurrent update: ~100-200MB
- Large models may require additional temporary space

CPU Usage:
- Minimal CPU usage (I/O bound operations)
- Most work done by Ollama process
- Async operations reduce blocking

Network Usage:
- Depends on model sizes and concurrent updates
- Bandwidth = (Model Size × Concurrent Updates) / Update Time
- Monitor network usage during peak downloads

Disk Usage:
- Temporary space during downloads
- Old models kept until new ones are verified
- Recommend 2x largest model size free space

OPTIMIZATION STRATEGIES:

1. Start with smaller models to test connectivity
2. Update during off-peak hours for better speeds
3. Use SSD storage for faster I/O operations
4. Monitor system resources during updates
5. Adjust concurrency based on system capabilities

================================================================================
                               SECURITY
================================================================================

SECURITY FEATURES:

1. No External Dependencies:
   - Uses only Python standard library
   - Reduces attack surface
   - No third-party package vulnerabilities

2. Minimal Privileges:
   - Runs with user permissions
   - No elevated privileges required
   - Respects system security policies

3. Safe Command Execution:
   - Subprocess calls are parameterized
   - No shell injection vulnerabilities
   - Input validation and sanitization

4. Data Protection:
   - No credential storage or transmission
   - Local operation only
   - Logs contain no sensitive information

5. Network Security:
   - Uses Ollama's official endpoints only
   - Respects proxy and firewall settings
   - No custom network protocols

SECURITY BEST PRACTICES:

1. Run with regular user account (not root/administrator)
2. Verify script integrity before execution
3. Monitor network traffic during updates
4. Keep Python and system updated
5. Review logs for unusual activity

PRIVACY CONSIDERATIONS:

- No personal data collection
- No telemetry or analytics
- Local operation only
- User consent for all operations

================================================================================
                              LIMITATIONS
================================================================================

KNOWN LIMITATIONS:

1. Network Dependency:
   - Requires stable internet connection
   - Large models need significant bandwidth
   - Network interruptions can cause failures

2. Platform Specific:
   - Requires Ollama to be properly installed
   - Command-line access required
   - Some features may vary between platforms

3. Resource Requirements:
   - Large models require substantial disk space
   - Concurrent updates increase memory usage
   - Limited by system capabilities

4. Model Compatibility:
   - Only works with Ollama-compatible models
   - Cannot update models from other sources
   - Respects Ollama's model format requirements

5. Concurrent Limitations:
   - High concurrency may overwhelm network
   - System resources limit practical concurrency
   - Ollama server may have rate limits

WORKAROUNDS:

1. Use dry-run mode to test before actual updates
2. Update in smaller batches for large collections
3. Monitor system resources during operation
4. Implement custom retry logic for failed updates
5. Schedule updates during off-peak hours

FUTURE ENHANCEMENTS:

- Resume interrupted downloads
- Custom model source support
- GUI interface option
- Integration with model management tools
- Advanced filtering and scheduling

================================================================================
                              DISCLAIMER
================================================================================

GENERAL DISCLAIMER:

This software is provided "AS IS" without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and non-infringement. In no event shall the
authors or copyright holders be liable for any claim, damages, or other
liability, whether in an action of contract, tort, or otherwise, arising from,
out of, or in connection with the software or the use or other dealings in the
software.

SPECIFIC DISCLAIMERS:

1. DATA LOSS:
   The authors are not responsible for any data loss, corruption, or system
   damage that may occur during the use of this software. Users should ensure
   adequate backups before running model updates.

2. NETWORK USAGE:
   This software may consume significant network bandwidth when updating large
   models. Users are responsible for any costs associated with network usage
   and should be aware of their internet service provider's data limits.

3. SYSTEM RESOURCES:
   The software may consume substantial system resources including disk space,
   memory, and processing power. Users should monitor system performance and
   ensure adequate resources are available.

4. THIRD-PARTY SERVICES:
   This software relies on Ollama and its associated services. The authors are
   not responsible for the availability, reliability, or functionality of these
   third-party services.

5. COMPATIBILITY:
   While designed to be cross-platform, the software may not work correctly on
   all systems or configurations. Users should test thoroughly in their
   specific environment.

6. SECURITY:
   Users are responsible for ensuring the security of their systems and should
   review the software's operation in the context of their security policies.

7. LEGAL COMPLIANCE:
   Users are responsible for ensuring their use of this software and any
   downloaded models complies with applicable laws and regulations in their
   jurisdiction.

LICENSE INFORMATION:

This software is released under the MIT License. See the LICENSE file for
complete terms and conditions.

SUPPORT DISCLAIMER:

This software is provided as-is with no guarantee of support. While the authors
may provide assistance on a best-effort basis, users should not expect
immediate or comprehensive support for issues encountered.

USE AT YOUR OWN RISK:

By using this software, you acknowledge that you have read and understood these
disclaimers and agree to use the software at your own risk.

================================================================================
                             VERSION HISTORY
================================================================================

Version 3.0 (Current - December 19, 2024):
- Complete rewrite for cross-platform compatibility
- Fixed subprocess.CREATE_NO_WINDOW issues on macOS/Linux
- Added async operations for concurrent updates
- Implemented interactive model selection
- Added comprehensive error handling and retry logic
- Created structured JSON reporting system
- Added dry-run mode for safe testing
- Implemented graceful signal handling
- Enhanced logging with multiple output options
- Optimized for Python 3.8+ with 3.13.5+ features

Version 2.0 (Previous):
- Added basic retry mechanisms
- Improved error messages
- Added progress tracking

Version 1.0 (Original):
- Basic model update functionality
- Windows-specific implementation
- Limited error handling

================================================================================
                              CONTRIBUTORS
================================================================================

Development Team:
- AI Test Case Generator Team

Special Thanks:
- Ollama development team for the excellent model management platform
- Python community for comprehensive standard library
- Open source contributors for inspiration and best practices

Contributing:
Contributions are welcome! Please ensure any pull requests include:
- Comprehensive testing across platforms
- Updated documentation
- Backward compatibility maintenance
- Security review for new features

Contact:
For questions, issues, or contributions, please refer to the project
repository or contact the development team.

================================================================================
                                 END
================================================================================

Thank you for using the Cross-Platform Ollama Model Updater!

For the latest updates and documentation, please check the project repository.

Last Updated: December 19, 2024
Version: 3.0 (Working)
================================================================================