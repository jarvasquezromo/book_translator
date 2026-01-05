"""PDF generation module using ReportLab.

This module handles:
- Bilingual PDF layout creation
- Image embedding with proper scaling
- Text rendering with position preservation
- Multi-page document support
"""

import io
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, darkgray
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .layout import LayoutConfiguration, TextBlock

logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generates bilingual PDF documents with images and text."""

    def __init__(self):
        """Initialize the PDF generator."""
        self.page_size = A4
        self.default_font = 'Helvetica'
        self.default_font_size = 10
        logger.info("PDFGenerator initialized")

    def create_bilingual_pdf(
        self,
        image: np.ndarray,
        layout_config: LayoutConfiguration,
        output_path: str
    ) -> None:
        """Create a bilingual PDF with original image and translations.

        Args:
            image: Original processed image.
            layout_config: Layout configuration with text blocks.
            output_path: Path for output PDF file.

        Raises:
            IOError: If PDF creation fails.
        """
        try:
            # Create PDF canvas
            pdf = canvas.Canvas(output_path, pagesize=self.page_size)

            # Set document metadata
            pdf.setTitle("Book Translation")
            pdf.setAuthor("Book Translator")
            pdf.setSubject("Bilingual Document")

            logger.info(f"Creating PDF: {output_path}")

            # Generate the page content
            self._render_page(pdf, image, layout_config)

            # Save PDF
            pdf.save()
            logger.info(f"PDF saved successfully: {output_path}")

        except Exception as e:
            logger.error(f"PDF generation failed: {e}", exc_info=True)
            raise IOError(f"Failed to create PDF: {e}")

    def _render_page(
        self,
        pdf: canvas.Canvas,
        image: np.ndarray,
        config: LayoutConfiguration
    ) -> None:
        """Render a single page with image and translations.

        Args:
            pdf: ReportLab canvas object.
            image: Original image to embed.
            config: Layout configuration.
        """
        # Draw page border (optional, for debugging)
        # self._draw_page_border(pdf, config)

        # Calculate image dimensions and position
        image_data = self._prepare_image_for_pdf(image)
        scale_factor = self._calculate_scale_factor(image, config)

        # Position image on left side
        img_x = config.margins['left']
        img_y = config.page_height - config.margins['top'] - (
            image.shape[0] * scale_factor
        )
        img_width = image.shape[1] * scale_factor
        img_height = image.shape[0] * scale_factor

        # Draw image
        pdf.drawInlineImage(
            image_data,
            img_x,
            img_y,
            width=img_width,
            height=img_height
        )

        logger.info(
            f"Image rendered at ({img_x:.1f}, {img_y:.1f}), "
            f"size: {img_width:.1f}x{img_height:.1f}"
        )

        # Draw vertical separator line
        self._draw_separator(pdf, config, img_width)

        # Render translated text blocks
        self._render_text_blocks(pdf, config, scale_factor)

    def _calculate_scale_factor(
        self,
        image: np.ndarray,
        config: LayoutConfiguration
    ) -> float:
        """Calculate scaling factor to fit image on page.

        Args:
            image: Input image.
            config: Layout configuration.

        Returns:
            Scale factor to apply.
        """
        # Available width for image (half the page minus margins)
        available_width = (
            (config.page_width - 
             config.margins['left'] - 
             config.margins['middle']) / 2
        )

        # Available height
        available_height = (
            config.page_height - 
            config.margins['top'] - 
            config.margins['bottom']
        )

        # Calculate scale factors for width and height
        width_scale = available_width / image.shape[1]
        height_scale = available_height / image.shape[0]

        # Use the smaller scale to fit both dimensions
        scale_factor = min(width_scale, height_scale)

        logger.info(f"Image scale factor: {scale_factor:.4f}")
        return scale_factor

    def _prepare_image_for_pdf(self, image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL Image for PDF embedding.

        Args:
            image: OpenCV image (BGR format).

        Returns:
            PIL Image object.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        return pil_image

    def _draw_separator(
        self,
        pdf: canvas.Canvas,
        config: LayoutConfiguration,
        img_width: float
    ) -> None:
        """Draw vertical separator between image and translation.

        Args:
            pdf: ReportLab canvas.
            config: Layout configuration.
            img_width: Width of the rendered image.
        """
        separator_x = (
            config.margins['left'] + 
            img_width + 
            config.margins['middle'] / 2
        )

        pdf.setStrokeColor(darkgray)
        pdf.setLineWidth(0.5)
        pdf.line(
            separator_x,
            config.margins['bottom'],
            separator_x,
            config.page_height - config.margins['top']
        )

    def _render_text_blocks(
        self,
        pdf: canvas.Canvas,
        config: LayoutConfiguration,
        scale_factor: float
    ) -> None:
        """Render all translated text blocks on PDF.

        Args:
            pdf: ReportLab canvas.
            config: Layout configuration with text blocks.
            scale_factor: Image scaling factor.
        """
        # Calculate translation section start position
        translation_x = (
            config.margins['left'] +
            config.image_width * scale_factor +
            config.margins['middle']
        )

        # Available width for text
        available_width = (
            config.page_width - 
            translation_x - 
            config.margins['right']
        )

        logger.info(f"Rendering {len(config.text_blocks)} text blocks")

        for i, block in enumerate(config.text_blocks):
            try:
                # Calculate position
                text_x = translation_x + (block.x * scale_factor * 0.1)
                # PDF coordinates are bottom-up, so we need to flip Y
                text_y = (
                    config.page_height - 
                    config.margins['top'] - 
                    (block.y * scale_factor)
                )

                # Ensure text stays within bounds
                text_x = max(translation_x, min(text_x, translation_x + available_width - 50))
                text_y = max(config.margins['bottom'], min(text_y, config.page_height - config.margins['top']))

                # Set font and size
                pdf.setFont(self.default_font, block.font_size)
                pdf.setFillColor(black)

                # Word wrap text to fit available width
                wrapped_text = self._wrap_text(
                    block.translated_text,
                    available_width,
                    block.font_size
                )

                # Draw each line
                line_height = block.font_size * block.line_spacing
                current_y = text_y

                for line in wrapped_text:
                    if current_y < config.margins['bottom']:
                        logger.warning(f"Text block {i} exceeds page bounds")
                        break

                    pdf.drawString(text_x, current_y, line)
                    current_y -= line_height

            except Exception as e:
                logger.error(f"Error rendering text block {i}: {e}")
                continue

    def _wrap_text(
        self,
        text: str,
        max_width: float,
        font_size: int
    ) -> list:
        """Wrap text to fit within specified width.

        Args:
            text: Text to wrap.
            max_width: Maximum width in points.
            font_size: Font size in points.

        Returns:
            List of text lines.
        """
        words = text.split()
        lines = []
        current_line = []

        # Rough estimate: average character width ≈ 0.5 * font_size
        char_width = font_size * 0.5
        max_chars = int(max_width / char_width)

        for word in words:
            # Check if adding this word would exceed the line length
            test_line = ' '.join(current_line + [word])

            if len(test_line) <= max_chars:
                current_line.append(word)
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        # Add last line
        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _draw_page_border(
        self,
        pdf: canvas.Canvas,
        config: LayoutConfiguration
    ) -> None:
        """Draw page border for debugging layout.

        Args:
            pdf: ReportLab canvas.
            config: Layout configuration.
        """
        pdf.setStrokeColor(darkgray)
        pdf.setLineWidth(0.5)
        pdf.rect(
            config.margins['left'],
            config.margins['bottom'],
            config.page_width - config.margins['left'] - config.margins['right'],
            config.page_height - config.margins['top'] - config.margins['bottom']
        )

    def add_header_footer(
        self,
        pdf: canvas.Canvas,
        page_num: int,
        total_pages: int,
        config: LayoutConfiguration
    ) -> None:
        """Add header and footer to page.

        Args:
            pdf: ReportLab canvas.
            page_num: Current page number.
            total_pages: Total number of pages.
            config: Layout configuration.
        """
        # Footer with page number
        pdf.setFont(self.default_font, 8)
        pdf.setFillColor(darkgray)

        footer_text = f"Page {page_num} of {total_pages}"
        text_width = pdf.stringWidth(footer_text, self.default_font, 8)

        footer_x = (config.page_width - text_width) / 2
        footer_y = config.margins['bottom'] - 20

        pdf.drawString(footer_x, footer_y, footer_text)
