"""
Pytest configuration and shared fixtures.

Provides reusable test data and utilities for all test modules.
"""

import pytest
import sys
from pathlib import Path

# Ensure src modules are importable in tests
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_docs():
    """Sample documents for testing - diverse compliance content."""
    return [
        """
        AML Customer Due Diligence Procedure v1
        
        This procedure outlines the verification of customer identity and beneficial ownership.
        KYC requirements mandate collection of government-issued identification documents.
        Customer screening against sanction lists is mandatory for all transactions.
        Risk assessment must be conducted before account activation and ongoing monitoring required.
        """,
        
        """
        Criminal Case Intake Procedure v1
        
        Law enforcement agencies must document all incoming criminal complaints and allegations.
        Initial investigation requires victim statement collection and evidence preservation.
        Case classification determines investigation priority and resource allocation.
        Documentation accuracy is critical for prosecution and legal admissibility in court.
        """,
        
        """
        BNS 2023 - Bharatiya Nyaya Sanhita
        
        The Bharatiya Nyaya Sanhita 2023 is the primary criminal law framework in India.
        It codifies all criminal offenses and provides punishment guidelines for violators.
        Section procedures dictate investigation, arrest, and prosecution methodologies.
        Compliance with BNS provisions is mandatory for law enforcement officers nationwide.
        """,
        
        """
        IT Act 2021 - Information Technology Cybercrime
        
        The Information Technology Act 2021 addresses cybersecurity threats and digital crimes.
        Cybercrime offense categories include hacking, malware distribution, and data theft.
        Reporting requirements mandate notification to CERT-In for data breach incidents.
        Organizations must implement cybersecurity measures per framework specifications.
        """,
    ]


@pytest.fixture
def sample_categories():
    """Category labels matching sample_docs fixture."""
    return ['Financial_Law', 'Criminal_Law', 'Criminal_Law', 'Cyber_Crime']


@pytest.fixture
def sample_small_corpus():
    """Minimal 3-document corpus for vectorization edge case testing."""
    return [
        "transaction monitoring compliance verification",
        "criminal offense investigation procedure",
        "cybersecurity incident response protocol"
    ]


@pytest.fixture
def sample_docs_single_category():
    """All documents in same category - should fail classification."""
    return [
        "AML transaction monitoring customer due diligence KYC verification",
        "Transaction monitoring sanction screening financial compliance",
        "Customer identification verification beneficial ownership assessment",
        "Ongoing transaction monitoring suspicious activity detection",
        "Transaction reporting threshold documentation compliance",
        "Financial transaction record keeping audit trail requirements",
    ]


@pytest.fixture
def sample_docs_imbalanced():
    """Imbalanced categories - one category has insufficient samples."""
    return [
        "Criminal law enforcement procedure investigation",
        "Criminal case documentation case file management",
        "Criminal prosecution evidence handling courtroom procedure",
        "Criminal law violation detection arrest procedure",
        "Cybersecurity incident response cyber attack mitigation",  # Only 1 - insufficient
    ]


@pytest.fixture
def sample_categories_imbalanced():
    """Categories matching imbalanced corpus."""
    return ['Criminal_Law', 'Criminal_Law', 'Criminal_Law', 'Criminal_Law', 'Cyber_Crime']


@pytest.fixture
def sample_categories_single():
    """Single category for all documents."""
    return ['Financial_Law'] * 6


@pytest.fixture
def empty_string():
    """Empty string for testing edge cases."""
    return ""


@pytest.fixture
def special_chars_text():
    """Text with special characters, numbers, and punctuation."""
    return """
    Section 3.14 (a)(i): Special chars !@#$%^&*()_+-=[]{}|;':",./<>?
    Numbers: 42 3.14 -100 0xFF 1e-5
    Mixed: test@example.com (123) 456-7890 https://example.com/path?query=value
    Emojis: 😀 ✓ ✗ ❌ ✅
    Unicode: café naïve résumé Москва
    """


@pytest.fixture
def special_chars_text_expected():
    """Expected output after preprocessing special_chars_text."""
    return "section 3.14 special chars numbers 42 3.14 100 0xff 1e 5 mixed test example com 123 456 7890 https example com path query value emojis unicode caf na v r sum moskva"
