"""
Setup & Environment Validation Script
======================================

Comprehensive diagnostics for TF-IDF Compliance Drift Detection System

This script:
1. Verifies project structure
2. Checks Python environment
3. Validates dependencies
4. Tests module imports
5. Checks data directory structure
6. Provides actionable recommendations
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


# ============================================================
# SECTION 1: PROJECT STRUCTURE VALIDATION
# ============================================================

def check_project_structure() -> Tuple[bool, List[str]]:
    """Verify all required directories and files exist"""
    print_header("SECTION 1: PROJECT STRUCTURE")
    
    issues = []
    root = Path(".")
    
    required_dirs = [
        "src",
        "data",
        "dashboard",
        "notebooks",
        "results",
        "scripts",
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "dashboard/app.py",
        ".gitignore",
    ]
    
    required_src_modules = [
        "src/alerts.py",
        "src/drift.py",
        "src/manual_tfidf_math.py",
        "src/preprocess.py",
        "src/similarity.py",
        "src/utils.py",
        "src/vectorize.py",
    ]
    
    required_data_dirs = [
        "data/guidelines",
        "data/internal",
        "data/guidelines_pdfs",
    ]
    
    all_checks = [
        ("Directories", required_dirs, "dir"),
        ("Core Files", required_files, "file"),
        ("Source Modules", required_src_modules, "file"),
        ("Data Directories", required_data_dirs, "dir"),
    ]
    
    for category, items, item_type in all_checks:
        print(f"\n{Colors.BOLD}{category}:{Colors.ENDC}")
        for item in items:
            path = root / item
            if item_type == "dir" and path.is_dir():
                print_success(item)
            elif item_type == "file" and path.is_file():
                print_success(item)
            else:
                print_error(f"Missing {item_type}: {item}")
                issues.append(f"Missing {item_type}: {item}")
    
    return len(issues) == 0, issues


# ============================================================
# SECTION 2: PYTHON ENVIRONMENT
# ============================================================

def check_python_environment() -> Tuple[bool, List[str]]:
    """Verify Python version and environment"""
    print_header("SECTION 2: PYTHON ENVIRONMENT")
    
    issues = []
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python Version: {Colors.BOLD}{py_version}{Colors.ENDC}")
    
    if sys.version_info >= (3, 8):
        print_success(f"Python {py_version} (supports 3.8+)")
    else:
        print_error(f"Python {py_version} (requires 3.8+)")
        issues.append(f"Python version too old: {py_version}")
    
    # Python executable
    print(f"Executable: {sys.executable}\n")
    
    # Virtual environment check
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print_success("Running in virtual environment")
    else:
        print_warning("Not in virtual environment (recommended to use one)")
    
    return len(issues) == 0, issues


# ============================================================
# SECTION 3: DEPENDENCY VALIDATION
# ============================================================

def check_dependencies() -> Tuple[bool, List[str]]:
    """Verify all required packages are installed"""
    print_header("SECTION 3: DEPENDENCIES")
    
    issues = []
    
    # Core dependencies
    core_packages = {
        'streamlit': 'UI Framework',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning (scikit-learn)',
        'matplotlib': 'Data visualization',
        'seaborn': 'Statistical visualization',
        'nltk': 'NLP preprocessing',
    }
    
    # Optional dependencies
    optional_packages = {
        'pdfplumber': 'PDF extraction',
        'fpdf': 'PDF generation',
        'wordcloud': 'Word cloud visualization',
        'pytesseract': 'OCR (requires Tesseract binary)',
        'pdf2image': 'PDF to image conversion',
    }
    
    print(f"{Colors.BOLD}Core Packages (Required):{Colors.ENDC}")
    for package, description in core_packages.items():
        try:
            importlib.import_module(package)
            print_success(f"{package:<20} - {description}")
        except ImportError:
            print_error(f"{package:<20} - {description}")
            issues.append(f"Missing required package: {package}")
    
    print(f"\n{Colors.BOLD}Optional Packages:{Colors.ENDC}")
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            print_success(f"{package:<20} - {description}")
        except ImportError:
            print_warning(f"{package:<20} - {description} (optional)")
    
    return len(issues) == 0, issues


# ============================================================
# SECTION 4: MODULE IMPORTS
# ============================================================

def check_module_imports() -> Tuple[bool, List[str]]:
    """Test importing all custom modules"""
    print_header("SECTION 4: MODULE IMPORTS")
    
    issues = []
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd() / "src"))
    
    modules_to_test = {
        'alerts': 'Drift alert generation',
        'drift': 'Compliance comparison engine',
        'manual_tfidf_math': 'Manual TF-IDF implementation',
        'preprocess': 'Text preprocessing',
        'similarity': 'Similarity calculations',
        'utils': 'Utility functions',
        'vectorize': 'TF-IDF vectorization',
    }
    
    print(f"{Colors.BOLD}Custom Modules:{Colors.ENDC}\n")
    for module_name, description in modules_to_test.items():
        try:
            module = importlib.import_module(module_name)
            print_success(f"{module_name:<25} - {description}")
        except Exception as e:
            print_error(f"{module_name:<25} - {description}")
            print(f"  Error: {str(e)[:80]}")
            issues.append(f"Cannot import {module_name}: {str(e)[:100]}")
    
    return len(issues) == 0, issues


# ============================================================
# SECTION 5: DATA DIRECTORY CHECK
# ============================================================

def check_data_structure() -> Tuple[bool, List[str]]:
    """Verify data directory contents"""
    print_header("SECTION 5: DATA STRUCTURE")
    
    issues = []
    data_root = Path("data")
    
    if not data_root.exists():
        print_error("data/ directory not found!")
        return False, ["data/ directory missing"]
    
    # Guidelines
    guidelines_dir = data_root / "guidelines"
    if guidelines_dir.exists():
        categories = [d for d in guidelines_dir.iterdir() if d.is_dir()]
        print(f"{Colors.BOLD}Guidelines Categories:{Colors.ENDC}")
        if categories:
            for cat in categories:
                files = list(cat.glob("*.txt"))
                file_count = len(files)
                if file_count > 0:
                    print_success(f"{cat.name:<20} - {file_count} file(s)")
                else:
                    print_warning(f"{cat.name:<20} - no .txt files")
        else:
            print_warning("No categories in guidelines/")
            issues.append("guidelines/ is empty - add Criminal_Law/, Cyber_Crime/, Financial_Law/")
    
    # Internal policies
    internal_dir = data_root / "internal"
    if internal_dir.exists():
        internal_files = list(internal_dir.glob("*.txt"))
        print(f"\n{Colors.BOLD}Internal Policies:{Colors.ENDC}")
        if internal_files:
            print_success(f"{len(internal_files)} policy file(s) found")
            for f in internal_files[:5]:
                print(f"  - {f.name}")
            if len(internal_files) > 5:
                print(f"  ... and {len(internal_files)-5} more")
        else:
            print_warning("No .txt files in internal/")
            issues.append("internal/ is empty - add sample policy documents")
    
    # Metadata
    metadata_file = data_root / "metadata.csv"
    if metadata_file.exists():
        print(f"\n{Colors.BOLD}Metadata:{Colors.ENDC}")
        print_success("metadata.csv found")
    else:
        print_warning("metadata.csv not found (optional)")
    
    return len(issues) == 0, issues


# ============================================================
# SECTION 6: STREAMLIT QUICK TEST
# ============================================================

def check_streamlit_setup() -> Tuple[bool, List[str]]:
    """Verify Streamlit can load the app"""
    print_header("SECTION 6: STREAMLIT SETUP")
    
    issues = []
    
    try:
        import streamlit as st
        print_success(f"Streamlit imported successfully")
        
        # Check if app.py exists
        if Path("dashboard/app.py").exists():
            print_success("dashboard/app.py found")
            
            # Provide instructions
            print(f"\n{Colors.BOLD}To run the dashboard:{Colors.ENDC}")
            print(f"  {Colors.CYAN}streamlit run dashboard/app.py{Colors.ENDC}")
            print(f"\nAccess at: {Colors.CYAN}http://localhost:8501{Colors.ENDC}")
        else:
            print_error("dashboard/app.py not found!")
            issues.append("dashboard/app.py missing")
    
    except ImportError as e:
        print_error(f"Cannot import Streamlit: {e}")
        issues.append("Streamlit not installed")
    
    return len(issues) == 0, issues


# ============================================================
# SECTION 7: RECOMMENDATIONS
# ============================================================

def generate_recommendations(all_issues: List[str]):
    """Provide actionable recommendations"""
    print_header("RECOMMENDATIONS & NEXT STEPS")
    
    if not all_issues:
        print_success("All checks passed! Your environment is ready.")
        print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
        print("1. Add sample regulatory documents to data/guidelines/")
        print("2. Add internal policies to data/internal/")
        print("3. Run: streamlit run dashboard/app.py")
        return
    
    print(f"{Colors.BOLD}Issues Found ({len(all_issues)}):{Colors.ENDC}\n")
    
    for issue in all_issues:
        print_error(issue)
    
    print(f"\n{Colors.BOLD}Recommended Actions:{Colors.ENDC}\n")
    
    # Group issues by type
    missing_packages = [i for i in all_issues if "Missing required package" in i]
    missing_files = [i for i in all_issues if "Missing" in i and "package" not in i]
    import_errors = [i for i in all_issues if "Cannot import" in i]
    data_issues = [i for i in all_issues if "data/" in i or "guidelines" in i]
    
    if missing_packages:
        print("Fix Missing Packages:")
        print(f"  Run: pip install -r requirements.txt")
    
    if missing_files:
        print("\nCreate Missing Files/Directories:")
        for issue in missing_files[:3]:
            print(f"  - {issue}")
    
    if import_errors:
        print("\nFix Import Errors:")
        print("  1. Verify src/ modules are valid Python files")
        print("  2. Check for syntax errors: python -m py_compile src/*.py")
        print("  3. Ensure __init__.py exists in src/ (if needed)")
    
    if data_issues:
        print("\nPopulate Data Directory:")
        print("  1. Add regulatory documents to data/guidelines/")
        print("  2. Add internal policies to data/internal/")
        print("  3. (Optional) Create data/metadata.csv with document labels")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   TF-IDF COMPLIANCE DRIFT DETECTION - SETUP VALIDATOR      ║")
    print("║         Environment & Dependency Verification              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    all_issues = []
    
    # Run all checks
    checks = [
        check_project_structure,
        check_python_environment,
        check_dependencies,
        check_module_imports,
        check_data_structure,
        check_streamlit_setup,
    ]
    
    for check_func in checks:
        try:
            success, issues = check_func()
            all_issues.extend(issues)
        except Exception as e:
            print_error(f"Error in {check_func.__name__}: {e}")
            all_issues.append(f"Error in {check_func.__name__}: {str(e)[:100]}")
    
    # Generate recommendations
    generate_recommendations(all_issues)
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    if not all_issues:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED - READY TO RUN!{Colors.ENDC}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ {len(all_issues)} ISSUE(S) FOUND{Colors.ENDC}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
