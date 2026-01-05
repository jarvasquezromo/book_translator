"""Main entry point for the book translation application.

This module orchestrates the entire translation workflow:
1. Image preprocessing and page detection
2. OCR text extraction with bounding boxes
3. Text translation
4. PDF generation with bilingual layout
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .vision import ImageProcessor
from .translator import TextTranslator
from .layout import LayoutManager
from .pdfgen import PDFGenerator


# Configure logging
logging.basicConfig(
    # level=logging.INFO,
    level=logging.DEBUG, #for debugging
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('book_translator.log')
    ]
)
logger = logging.getLogger(__name__)


class BookTranslator:
    """Orchestrates the complete book translation pipeline."""

    def __init__(self, source_lang: str = 'auto', target_lang: str = 'en'):
        """Initialize the book translator.

        Args:
            source_lang: Source language code (default: 'auto' for auto-detection).
            target_lang: Target language code for translation.
        """
        self.image_processor = ImageProcessor()
        self.translator = TextTranslator(source_lang, target_lang)
        self.layout_manager = LayoutManager()
        self.pdf_generator = PDFGenerator()
        logger.info(
            f"BookTranslator initialized: {source_lang} -> {target_lang}"
        )

    def translate_book_page(
        self,
        image_path: Path,
        output_path: Path,
        dpi: int = 300
    ) -> bool:
        """Translate a single book page and generate PDF output.

        Args:
            image_path: Path to the input image file.
            output_path: Path for the output PDF file.
            dpi: DPI resolution for OCR processing.

        Returns:
            True if translation succeeds, False otherwise.

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If image processing fails.
        """
        try:
            # Step 1: Validate input
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            logger.info(f"Processing image: {image_path}")

            # Step 2: Image preprocessing
            processed_image = self.image_processor.process_page(
                str(image_path)
            )
            if processed_image is None:
                raise ValueError("Image preprocessing failed")

            # Step 3: OCR text extraction
            logger.info("Extracting text with OCR...")
            ocr_results = self.image_processor.extract_text_with_boxes(
                processed_image,
                dpi=dpi
            )

            if not ocr_results:
                logger.warning("No text detected in image")
                return False

            logger.info(f"Detected {len(ocr_results)} text blocks")

            # Step 4: Translate text blocks
            logger.info("Translating text blocks...")
            translated_blocks = []
            for block in ocr_results:
                translated_text = self.translator.translate(block['text'])
                translated_blocks.append({
                    'original': block['text'],
                    'translated': translated_text,
                    'bbox': block['bbox'],
                    'confidence': block['confidence']
                })

            # Step 5: Generate layout
            logger.info("Generating bilingual layout...")
            layout_config = self.layout_manager.create_layout(
                processed_image,
                translated_blocks
            )

            # Step 6: Create PDF
            logger.info(f"Generating PDF: {output_path}")
            self.pdf_generator.create_bilingual_pdf(
                processed_image,
                layout_config,
                str(output_path)
            )

            logger.info("Translation completed successfully!")
            return True

        except FileNotFoundError as e:
            logger.error(f"File error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Processing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return False


def main() -> int:
    """Main function with CLI argument parsing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description='Translate book pages from images to PDF'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input image file'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Path to output PDF file'
    )
    parser.add_argument(
        '--source-lang',
        type=str,
        default='auto',
        help='Source language code (default: auto-detect)'
    )
    parser.add_argument(
        '--target-lang',
        type=str,
        default='en',
        help='Target language code (default: en)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for OCR processing (default: 300)'
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        translator = BookTranslator(args.source_lang, args.target_lang)
        success = translator.translate_book_page(
            input_path,
            output_path,
            args.dpi
        )

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
