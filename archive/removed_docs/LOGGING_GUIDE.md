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

### For Frontend (Streamlit) Modules

The frontend (`frontend/app.py`) initializes logging automatically:

```python
from utils.logging_setup import setup_logging
import logging

def main():
    # Initialize logging at application startup
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Now logging works throughout the app
    logger.info("App started")
```

## Configuration

### Default Setup

```python
from utils.logging_setup import setup_logging

# Uses default: INFO level, logs to stdout only
setup_logging()
```

### Custom Configuration

```python
from utils.logging_setup import setup_logging
import logging

# Enable DEBUG-level logging and write to file
setup_logging(
    level=logging.DEBUG,
    log_file="compliance_drift.log",
    format_string="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
```

## Log Levels

| Level | Purpose | Example |
|-------|---------|---------|
| **DEBUG** | Detailed diagnostic info | `logger.debug("Cache hits: %d", hits)` |
| **INFO** | General informational messages | `logger.info("Documents loaded: %d", count)` |
| **WARNING** | Warning about potential issues | `logger.warning("Vectorization degraded: %s", reason)` |
| **ERROR** | Error occurred but didn't crash | `logger.error("Classification failed: %s", error)` |
| **CRITICAL** | Critical error, likely crash | `logger.critical("Out of memory!")` |

## Usage Examples

### Backend: TF-IDF Engine

```python
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def build_tfidf_vectors(...):
    logger.info("Building TF-IDF vectors for %d docs", len(docs))
    
    try:
        vectorizer = TfidfVectorizer(...)
        vectors = vectorizer.fit_transform(docs)
        logger.info("Vectorization complete: %d features", vectors.shape[1])
        return vectorizer, vectors
    except ValueError as e:
        logger.error("Vectorization failed: %s", e)
        raise
```

### Backend: Classification

```python
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def perform_classification(...):
    logger.info("Classification: %d docs, %d classes", len(docs), len(set(categories)))
    
    if min_class_count < 3:
        logger.warning("Class imbalance detected: %d samples in minority class", min_class_count)
        return {"warning": "Insufficient samples per class"}
    
    logger.info("Training classifier...")
    # ... training logic
```

### Frontend: Document Upload

```python
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def upload_documents(cfg):
    logger.info("Starting document upload")
    docs = []
    
    for file in uploaded_files:
        try:
            doc = load_document(file)
            docs.append(doc)
            logger.debug("Loaded: %s (%d bytes)", file.name, len(file.getvalue()))
        except Exception as e:
            logger.error("Failed to load %s: %s", file.name, e)
    
    logger.info("Upload complete: %d documents loaded", len(docs))
    return docs
```

## Viewing Logs

### Streamlit Console Output

When running the Streamlit app, all logs are printed to the terminal:

```
$ streamlit run frontend/app.py
2026-02-07 10:23:45,123 - frontend.app - INFO - App started
2026-02-07 10:23:47,456 - frontend.components.file_upload - INFO - Documents loaded: 5 internal, 3 guidelines
2026-02-07 10:23:48,789 - backend.tfidf_engine - INFO - Building TF-IDF vectors for 8 docs
2026-02-07 10:23:49,012 - backend.tfidf_engine - INFO - Vectorization complete: 5000 features
```

### File Logging

To capture logs to a file:

```python
setup_logging(
    level=logging.INFO,
    log_file="logs/compliance_drift.log"
)
```

Then view logs:

```bash
# Watch logs in real-time
tail -f logs/compliance_drift.log

# Search for errors
grep ERROR logs/compliance_drift.log

# Count log levels
grep -c WARNING logs/compliance_drift.log
```

## Best Practices

1. **Use appropriate levels:**
   - `DEBUG` for detailed diagnostic info (variable values, loop iterations)
   - `INFO` for significant events (load, start, complete)
   - `WARNING` for potential issues (degraded mode, insufficient data)
   - `ERROR` for failures (exceptions caught and handled)
   - `CRITICAL` for catastrophic failures (out of memory)

2. **Include context:**
   ```python
   # ❌ Bad
   logger.info("Processing")
   
   # ✅ Good
   logger.info("Processing %d documents with %d features", len(docs), features)
   ```

3. **Use exception logging:**
   ```python
   # ❌ Just log the exception
   logger.error("Error: %s", str(e))
   
   # ✅ Use logger.exception to include traceback
   logger.exception("Processing failed for document %s", filename)
   ```

4. **Avoid logging sensitive data:**
   ```python
   # ❌ Bad - exposes document content
   logger.info("Document text: %s", full_text[:200])
   
   # ✅ Good - logs only metadata
   logger.info("Processing document: %s (%d chars)", filename, len(full_text))
   ```

## Migration Notes

### Existing Modules

Existing modules (like `backend/text_processing.py`) that use:

```python
logger = logging.getLogger(__name__)
```

Will continue to work without modification. The `setup_logging()` call in `frontend/app.py` configures the entire logging system.

**Optional:** Update existing modules to use the centralized function:

```python
# Before
import logging
logger = logging.getLogger(__name__)

# After
from utils.logging_setup import get_logger
logger = get_logger(__name__)
```

Both approaches work identically. Use the second for consistency.

## Troubleshooting

### Logs not appearing?

1. Check that `setup_logging()` is called before your code logs
2. Verify the logging level is not too high (e.g., WARNING won't show INFO messages)
3. If using file logging, check permissions and disk space

### Too much output?

Set logging level to WARNING or ERROR:

```python
setup_logging(level=logging.WARNING)
```

Or use environment variable:

```python
import os
level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
setup_logging(level=level)
```

### Log files getting too large?

Use Python's `RotatingFileHandler` for automatic log rotation:

```python
# In a future enhancement to logging_setup.py
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "compliance_drift.log",
    maxBytes=10_000_000,  # 10 MB
    backupCount=5         # Keep 5 historical logs
)
```
