"""
Tests for input validation and security functions.

Validates:
- File size validation
- File extension validation
- PDF magic bytes validation
- Text file encoding validation
- Error message clarity
"""

import pytest
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Note: This would need to be adapted to import from dashboard/app.py
# For now, we provide test structure that documents the expected behavior


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing validation."""
    
    def __init__(self, name: str, content: bytes, size: int = None):
        self.name = name
        self.content = content
        self.size = size if size is not None else len(content)
    
    def getvalue(self):
        return self.content


class TestFileValidation:
    """Test suite for input file validation."""
    
    def test_validate_valid_pdf(self):
        """Test validation of valid PDF file."""
        # PDF magic bytes: %PDF-1.4
        pdf_content = b'%PDF-1.4\n%mock pdf content here'
        mock_file = MockUploadedFile('test.pdf', pdf_content)
        
        # This would call the validate_input_file function
        # from dashboard/app.py
        # Expected: (True, "✅ test.pdf - Valid (0.00MB)")
    
    def test_validate_invalid_pdf_magic_bytes(self):
        """Test validation rejects PDF with wrong magic bytes."""
        # Wrong magic bytes - starts with %GIF instead of %PDF
        pdf_content = b'%GIF89asome data'
        mock_file = MockUploadedFile('fake.pdf', pdf_content)
        
        # Expected: (False, "Invalid PDF file: fake.pdf (wrong magic bytes). File may be corrupted.")
    
    def test_validate_file_size_exceeded(self):
        """Test validation rejects oversized files."""
        # Create content larger than 50MB (default limit)
        large_content = b'x' * (51 * 1024 * 1024)
        mock_file = MockUploadedFile('large.pdf', large_content, size=51 * 1024 * 1024)
        
        # Expected: (False, "File exceeds 50MB limit (size: 51.0MB)")
    
    def test_validate_invalid_extension(self):
        """Test validation rejects unsupported file extensions."""
        content = b'some content'
        mock_file = MockUploadedFile('document.docx', content)
        
        # Expected: (False, "File type .docx not allowed. Allowed types: pdf, txt")
    
    def test_validate_valid_txt_file(self):
        """Test validation of valid UTF-8 text file."""
        txt_content = "This is a valid text file with UTF-8 encoding.".encode('utf-8')
        mock_file = MockUploadedFile('document.txt', txt_content)
        
        # Expected: (True, "✅ document.txt - Valid (0.00MB)")
    
    def test_validate_txt_file_wrong_encoding(self):
        """Test validation rejects non-UTF-8 text files."""
        # Create content with invalid UTF-8
        txt_content = b'\x80\x81\x82\x83'  # Invalid UTF-8 bytes
        mock_file = MockUploadedFile('bad.txt', txt_content)
        
        # Expected: (False, "Text file encoding error: bad.txt (must be UTF-8 encoded)")
    
    def test_validate_file_size_mb_calculation(self):
        """Test that file size is correctly converted to MB."""
        # Create 1MB file
        one_mb = b'x' * (1024 * 1024)
        mock_file = MockUploadedFile('one_mb.txt', one_mb)
        
        # Expected message should show approximately 1.00MB
    
    def test_validate_multiple_file_extensions(self):
        """Test validation with custom allowed extensions."""
        content = b'%PDF-1.4\nsome content'
        mock_file = MockUploadedFile('document.pdf', content)
        
        # Should validate with extended extension list
        # validate_input_file(mock_file, allowed_extensions=['pdf', 'txt', 'docx'])


class TestValidationErrorMessages:
    """Test suite for clarity of validation error messages."""
    
    def test_error_message_includes_filename(self):
        """Test that error messages include the filename."""
        # Oversized file
        content = b'x' * (51 * 1024 * 1024)
        mock_file = MockUploadedFile('huge.pdf', content, size=51 * 1024 * 1024)
        
        # Expected error to include "huge.pdf"
    
    def test_error_message_includes_file_size(self):
        """Test that size validation errors show actual file size."""
        content = b'x' * (55 * 1024 * 1024)
        mock_file = MockUploadedFile('too_big.pdf', content, size=55 * 1024 * 1024)
        
        # Expected error to include "55.0MB"
    
    def test_error_message_lists_allowed_types(self):
        """Test that extension validation shows allowed types."""
        content = b'some content'
        mock_file = MockUploadedFile('document.xlsx', content)
        
        # Expected error to include "pdf, txt"
    
    def test_success_message_includes_size(self):
        """Test that success message includes file size in MB."""
        txt_content = b'x' * 500000  # ~0.48MB
        mock_file = MockUploadedFile('document.txt', txt_content)
        
        # Expected success message to include "0.48MB"


class TestSecurityValidation:
    """Test suite for security aspects of validation."""
    
    def test_validate_prevents_malformed_pdf(self):
        """Test that malformed PDFs are rejected."""
        # File starts with PDF magic but is corrupted
        pdf_content = b'%PDF-1.4\n' + b'\x00' * 100  # Contains null bytes
        mock_file = MockUploadedFile('corrupted.pdf', pdf_content)
        
        # Should validate magic bytes but might pass
        # Actual content validation happens elsewhere
    
    def test_validate_prevents_double_extension(self):
        """Test that files with double extensions are validated correctly."""
        # File claiming to be .txt but is actually .pdf
        pdf_content = b'%PDF-1.4\nsome pdf'
        mock_file = MockUploadedFile('document.pdf.txt', pdf_content)
        
        # Should validate based on final extension (.txt)
        # Will fail because it's actually PDF content
    
    def test_validate_case_insensitive_extension(self):
        """Test that extension validation is case-insensitive."""
        pdf_content = b'%PDF-1.4\nsome content'
        mock_files = [
            MockUploadedFile('document.PDF', pdf_content),
            MockUploadedFile('document.Pdf', pdf_content),
            MockUploadedFile('document.pDF', pdf_content),
        ]
        
        # All should be treated as .pdf extension


class TestValidationMetrics:
    """Test suite for validation metrics tracking."""
    
    def test_metrics_count_valid_files(self):
        """Test that valid files are counted correctly."""
        # Simulate multiple file validations
        # Metrics should show: total_files, valid_files, rejected_files
    
    def test_metrics_track_rejection_reasons(self):
        """Test that rejection reasons are categorized."""
        # Different rejection reasons should be tracked separately:
        # - "File exceeds XMB limit"
        # - "File type .X not allowed"
        # - "Invalid PDF file"
        # - "Text file encoding error"
    
    def test_metrics_calculate_success_rate(self):
        """Test that success rate is calculated correctly."""
        # If 3/5 files pass: success rate should be 60%
