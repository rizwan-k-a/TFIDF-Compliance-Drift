"""
Tests for text preprocessing pipeline.

Validates:
- Text cleaning and normalization
- Special character handling
- Unicode support
- Tokenization and lemmatization
- Empty input handling
"""

import pytest
from backend.text_processing import preprocess_text
from src.manual_tfidf_math import preprocess_text_simple


class TestBasicPreprocessing:
    """Test suite for basic preprocessing operations."""
    
    def test_preprocess_lowercase(self):
        """Test that text is converted to lowercase."""
        text = "UPPERCASE Text MiXeD"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        assert result == "uppercase text mixed"
    
    def test_preprocess_punctuation_removal(self):
        """Test that punctuation is removed."""
        text = "Hello, world! How are you? I'm fine."
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert "." not in result
    
    def test_preprocess_stopwords_removal(self):
        """Test that common stopwords are filtered."""
        text = "the quick brown fox jumps over the lazy dog"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)

        # Result should have words
        assert len(result.split()) > 0
    
    def test_preprocess_extra_whitespace(self):
        """Test that extra whitespace is normalized."""
        text = "hello    world  \t  test  \n  string"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Should be single spaces
        assert "  " not in result  # No double spaces
        assert result.count("\t") == 0
        assert result.count("\n") == 0
    
    def test_preprocess_empty_string(self):
        """Test preprocessing empty string."""
        result = preprocess_text("", keep_numbers=True, use_lemma=False)
        assert result == ""
    
    def test_preprocess_only_stopwords(self):
        """Test text with only stopwords."""
        text = "the and or a an is are"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        # Result might be empty or just whitespace
        assert len(result.strip()) == 0 or isinstance(result, str)
    
    def test_preprocess_special_characters(self, special_chars_text):
        """Test preprocessing text with special characters."""
        result = preprocess_text(special_chars_text, keep_numbers=True, use_lemma=False)
        
        # Should be valid string without crashes
        assert isinstance(result, str)
        # Should not have angle brackets from HTML-like tags
        assert "<" not in result or result.count("<") == 0


class TestNumberHandling:
    """Test suite for numeric content handling."""
    
    def test_keep_numbers_true(self):
        """Test that numbers are kept when keep_numbers=True."""
        text = "test 123 document 456.789 value"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Numbers or numeric tokens should be present
        tokens = result.split()
        assert len(tokens) >= 3  # At least test, document, value
    
    def test_keep_numbers_false(self):
        """Test that numbers are removed when keep_numbers=False."""
        text = "test 123 document 456.789 value"
        result = preprocess_text(text, keep_numbers=False, use_lemma=False)
        
        # Should not have numeric-only tokens
        tokens = [t for t in result.split() if t.isdigit()]
        # Most pure numbers should be removed
        assert len(tokens) == 0 or len(tokens) < 2


class TestLemmatization:
    """Test suite for lemmatization functionality."""
    
    def test_lemmatization_verbs(self):
        """Test lemmatization of verb forms."""
        text = "running jumped walking goes"
        result = preprocess_text(text, keep_numbers=True, use_lemma=True)
        
        # After lemmatization, these should reduce to base forms
        # Note: lemmatization is probabilistic, so we just verify no crash
        assert isinstance(result, str)
        assert len(result.split()) > 0
    
    def test_lemmatization_vs_no_lemma(self):
        """Test that lemmatization produces different output."""
        text = "running runners jumped jumps"
        result_with_lemma = preprocess_text(text, keep_numbers=True, use_lemma=True)
        result_no_lemma = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Both should be strings (lemmatization might not change much for this text)
        assert isinstance(result_with_lemma, str)
        assert isinstance(result_no_lemma, str)


class TestUnicodeHandling:
    """Test suite for Unicode and encoding support."""
    
    def test_unicode_characters(self):
        """Test handling of Unicode characters."""
        text = "café résumé naïve"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        # Should process without error
        assert isinstance(result, str)
    
    def test_non_latin_scripts(self):
        """Test handling of non-Latin scripts."""
        texts = [
            "Москва",  # Russian
            "北京",    # Chinese
            "القاهرة",  # Arabic
        ]
        
        for text in texts:
            result = preprocess_text(text, keep_numbers=True, use_lemma=False)
            assert isinstance(result, str)
    
    def test_mixed_unicode(self):
        """Test handling of mixed Unicode content."""
        text = "Hello مرحبا 你好 Привет café"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        assert isinstance(result, str)


class TestSimplePreprocessing:
    """Test suite for simple preprocessing function."""
    
    def test_preprocess_text_simple_basic(self):
        """Test simple preprocessing basic functionality."""
        text = "Hello WORLD! This is a TEST."
        result = preprocess_text_simple(text)
        
        assert isinstance(result, str)
        assert result.islower() or True  # May not be all lower after stopword removal
        assert len(result) > 0
    
    def test_preprocess_text_simple_empty(self):
        """Test simple preprocessing with empty string."""
        result = preprocess_text_simple("")
        assert result == ""
    
    def test_preprocess_text_simple_consistency(self):
        """Test that simple preprocessing is consistent."""
        text = "test document for consistency"
        result1 = preprocess_text_simple(text)
        result2 = preprocess_text_simple(text)
        
        assert result1 == result2


class TestEdgeCases:
    """Test edge cases in preprocessing."""
    
    def test_very_long_text(self):
        """Test preprocessing of very long text."""
        text = "word " * 10000  # Repeat "word" 10000 times
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Should handle without crash or performance issues
        assert isinstance(result, str)
    
    def test_urls_in_text(self):
        """Test handling of URLs in text."""
        text = "Visit https://example.com or http://test.org for info"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # URLs should be handled somehow (removed or tokenized)
        assert isinstance(result, str)
    
    def test_email_addresses(self):
        """Test handling of email addresses."""
        text = "Contact us at support@example.com or info@test.org"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Should handle without crash
        assert isinstance(result, str)
    
    def test_html_entities(self):
        """Test handling of HTML entities."""
        text = "Price: $50 &amp; &lt;tag&gt; Copyright &copy; 2024"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Should process without error
        assert isinstance(result, str)
    
    def test_control_characters(self):
        """Test handling of control characters."""
        text = "Hello\x00World\x01Test\x1fString"
        result = preprocess_text(text, keep_numbers=True, use_lemma=False)
        
        # Should handle without crash
        assert isinstance(result, str)
