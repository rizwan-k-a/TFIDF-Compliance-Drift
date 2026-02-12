````markdown
# Security Hardening: File Validation

## Overview

File upload validation is a critical security layer. The enhanced `validate_input_file()` function in [backend/utils.py](backend/utils.py) implements **Defense in Depth** with 6 layers of protection.

---

## Vulnerabilities Fixed

### 1. ❌ Path Traversal (CRITICAL)

**Before:**
```python
# Accepts filenames like:
"../../../etc/passwd"
"\\..\\..\\windows\\system32"
"/etc/hosts"
```

**Attack:** Attacker uploads file with path traversal to read/write system files.

**After:**
```python
# Rejects all attempts:
if ".." in name or name.startswith("/") or name.startswith("\\"):
    return FileValidationResult(False, reason="Invalid filename: path traversal detected")
```

**Impact:** ✅ Prevents directory traversal attacks

...[truncated for archive]

````
