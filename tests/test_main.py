"""Tests for the book translator main module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.main import BookTranslator


class TestBookTranslator:
    """Test the BookTranslator class."""

    def test_initialization(self):
        """Test that BookTranslator initializes correctly."""
        translator = BookTranslator('es', 'en')
        assert translator.source_lang == 'es'
        assert translator.target_lang == 'en'

    @patch('src.main.ImageProcessor')
    @patch('src.main.TextTranslator')
    @patch('src.main.LayoutManager')
    @patch('src.main.PDFGenerator')
    def test_translate_book_page_success(self, mock_pdf, mock_layout, mock_translator, mock_vision):
        """Test successful translation of a book page."""
        # Setup mocks
        mock_vision_instance = Mock()
        mock_vision.return_value = mock_vision_instance
        mock_vision_instance.process_page.return_value = Mock()
        mock_vision_instance.extract_text_with_boxes.return_value = [
            {'text': 'Hola', 'bbox': [0, 0, 100, 50], 'confidence': 90}
        ]

        mock_translator_instance = Mock()
        mock_translator.return_value = mock_translator_instance
        mock_translator_instance.translate.return_value = 'Hello'

        mock_layout_instance = Mock()
        mock_layout.return_value = mock_layout_instance
        mock_layout_instance.create_layout.return_value = Mock()

        mock_pdf_instance = Mock()
        mock_pdf.return_value = mock_pdf_instance

        # Test
        translator = BookTranslator()
        input_path = Path('test.jpg')
        output_path = Path('output.pdf')

        # Mock file existence
        with patch.object(input_path, 'exists', return_value=True):
            result = translator.translate_book_page(input_path, output_path)

        assert result is True
        mock_pdf_instance.create_bilingual_pdf.assert_called_once()