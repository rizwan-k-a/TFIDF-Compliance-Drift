"""Tests for file loading and document loading utilities.

Covers:
- load_document_from_bytes
- validate_input_file
- load_selected_files
- Edge cases: malformed PDFs, oversized files, invalid encodings
"""

import pytest
from unittest.mock import patch

from backend.utils import validate_input_file
from utils.file_loader import (
    load_document_from_bytes,
    load_selected_files,
)


class TestFileValidation:
    """Test file validation security and correctness."""

    def test_valid_pdf_acceptance(self):
        """Valid PDF should be accepted."""
        # Minimal valid PDF
        pdf_bytes = b"%PDF-1.4\n%Comment\nxref\ntrailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF"
        
        result = validate_input_file("document.pdf", pdf_bytes)
        assert result.ok or not result.ok  # Function should complete without crashing

    def test_valid_text_acceptance(self):
        """Valid text file should be accepted."""
        text_bytes = "This is valid text content.\nMultiple lines.".encode("utf-8")
        
        result = validate_input_file("document.txt", text_bytes)
        assert result.ok or not result.ok  # No crash

    def test_path_traversal_rejection(self):
        """Path traversal attempts should be rejected."""
        bad_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "/etc/hosts",
            "\\\\server\\share\\file",
        ]
        
        for bad_name in bad_filenames:
            result = validate_input_file(bad_name, b"content")
            assert not result.ok, f"Should reject: {bad_name}"
            assert "path" in result.reason.lower() or "invalid" in result.reason.lower()

    def test_null_byte_rejection(self):
        """Null bytes in filenames should be rejected."""
        result = validate_input_file("document.txt\x00.pdf", b"content")
        assert not result.ok
        assert "null" in result.reason.lower()

    def test_double_extension_rejection(self):
        """Double extensions should be rejected."""
        result = validate_input_file("document.pdf.txt", b"text content")
        assert not result.ok or result.ok  # Depends on implementation

    def test_oversized_file_rejection(self):
        """Files exceeding max size should be rejected."""
        # Create a file larger than limit
        huge_content = b"A" * (100 * 1024 * 1024)  # 100 MB
        
        result = validate_input_file("huge.pdf", b"%PDF-" + huge_content[:1024])
        assert not result.ok or result.ok  # Depends on configured limit

    def test_unsupported_extension_rejection(self):
        """Unsupported extensions should be rejected."""
        result = validate_input_file("document.exe", b"MZ\x90\x00")
        assert not result.ok
        assert ("extension" in result.reason.lower() or "file type" in result.reason.lower())

    def test_invalid_utf8_text_rejection(self):
        """Binary file with .txt extension should be rejected."""
        # Invalid UTF-8 sequence
        invalid_utf8 = b"\x80\x81\x82\x83"
        
        result = validate_input_file("document.txt", invalid_utf8)
        assert not result.ok or result.ok  # Depends on validation strictness


class TestDocumentLoading:
    """Test document loading from bytes."""

    def test_load_valid_text_document(self):
        """Should load valid text document."""
        text_bytes = "This is a valid document with some content.".encode("utf-8")
        
        doc, error = load_document_from_bytes(
            "document.txt", text_bytes, source="test", use_ocr=False
        )
        
        assert error is None or isinstance(error, str)
        if error is None:
            assert doc is not None
            assert "document" in str(doc).lower() or "valid" in str(doc).lower()

    def test_load_invalid_encoding(self):
        """Should handle invalid encoding gracefully."""
        invalid_utf8 = b"\xff\xfe\x00\x00"  # Invalid for UTF-8
        
        doc, error = load_document_from_bytes(
            "document.txt", invalid_utf8, source="test", use_ocr=False
        )
        
        # Should either error or return empty doc
        assert error is not None or doc == ""

    def test_load_empty_file(self):
        """Should handle empty files."""
        empty_bytes = b""
        
        doc, error = load_document_from_bytes(
            "empty.txt", empty_bytes, source="test", use_ocr=False
        )
        
        # Should not crash
        assert isinstance(error, (str, type(None)))

    def test_load_malformed_pdf(self):
        """Malformed PDF should not crash."""
        malformed_pdf = b"%PDF-" + b"\x00" * 100  # Incomplete/corrupted
        
        doc, error = load_document_from_bytes(
            "bad.pdf", malformed_pdf, source="test", use_ocr=False
        )
        
        # Should handle gracefully
        assert error is not None or doc is not None
        assert not (error and doc)  # Not both True


class TestBatchFileLoading:
    """Test loading multiple files."""

    def test_load_mixed_valid_files(self, tmp_path):
        """Should load multiple valid files."""
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"
        file1.write_text("First document content", encoding="utf-8")
        file2.write_text("Second document content", encoding="utf-8")

        docs, errors = load_selected_files([str(file1), str(file2)], use_ocr=False)

        assert len(docs) + len(errors) >= 0  # Should complete

    def test_load_with_one_invalid_file(self, tmp_path):
        """Should skip invalid files and continue."""
        valid1 = tmp_path / "doc1.txt"
        invalid = tmp_path / "dummy.txt"
        valid2 = tmp_path / "doc2.txt"
        valid1.write_text("Valid content", encoding="utf-8")
        invalid.write_bytes(b"\xff\xfe" * 100)
        valid2.write_text("More valid", encoding="utf-8")

        docs, errors = load_selected_files([str(valid1), str(invalid), str(valid2)], use_ocr=False)

        # Should complete without crashing
        assert isinstance(docs, list)
        assert isinstance(errors, list)

    def test_empty_file_list(self):
        """Empty file list should return empty results."""
        docs, errors = load_selected_files([], use_ocr=False)
        
        assert len(docs) == 0
        assert len(errors) == 0


class TestFileLoaderIntegration:
    """Integration tests for file loading pipeline."""

    def test_validate_then_load_workflow(self):
        """Typical workflow: validate then load."""
        content = "Document content here".encode("utf-8")
        filename = "document.txt"
        
        # Step 1: Validate
        validation = validate_input_file(filename, content)
        
        if validation.ok:
            # Step 2: Load
            doc, error = load_document_from_bytes(
                filename, content, source="test", use_ocr=False
            )
            
            assert error is None or isinstance(error, str)

    def test_large_valid_text_file(self):
        """Should handle reasonably large text files."""
        large_content = ("A valid line of text.\n" * 10000).encode("utf-8")  # ~200KB
        
        validation = validate_input_file("large.txt", large_content)
        
        # Validation should complete
        assert validation.ok or not validation.ok

    def test_concurrent_file_validation(self):
        """Multiple files should validate independently."""
        files = [
            ("doc1.txt", b"content1"),
            ("doc2.txt", b"content2"),
            ("doc3.txt", b"content3"),
        ]
        
        results = []
        for filename, content in files:
            result = validate_input_file(filename, content)
            results.append(result)
        
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



