````markdown
# Logging Setup Guide

## Overview

The application now includes centralized logging configuration via `utils/logging_setup.py`. This enables structured logging across all backend, frontend, and utility modules without requiring complex configuration.

## Quick Start

### For Backend Modules (no Streamlit imports)

```python
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Use logger in your code:
logger.debug("Detailed diagnostic information")
logger.info("General information messages")
logger.warning("Warning about potential issues")
logger.error("Error that occurred, but didn't crash")
logger.critical("Critical error that may cause crash")
```

...[truncated for archive]

````
