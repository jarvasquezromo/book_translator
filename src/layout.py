"""Layout management module for bilingual PDF generation.

This module handles:
- Spatial analysis of text blocks
- Layout optimization for readability
- Coordinate mapping between original and translated text
- Page composition strategy
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a text block with position and content."""
    original_text: str
    translated_text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    font_size: int = 12
    line_spacing: float = 1.2

    @property
    def center_x(self) -> int:
        """Get horizontal center coordinate."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Get vertical center coordinate."""
        return self.y + self.height // 2

    @property
    def bottom(self) -> int:
        """Get bottom coordinate."""
        return self.y + self.height

    @property
    def right(self) -> int:
        """Get right coordinate."""
        return self.x + self.width


@dataclass
class LayoutConfiguration:
    """Configuration for bilingual PDF layout."""
    image_width: int
    image_height: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    layout_mode: str = 'side-by-side'  # 'side-by-side' or 'overlay'
    margins: Dict[str, int] = field(default_factory=lambda: {
        'top': 50,
        'bottom': 50,
        'left': 50,
        'right': 50,
        'middle': 30
    })
    page_width: int = 595  # A4 width in points (72 DPI)
    page_height: int = 842  # A4 height in points


class LayoutManager:
    """Manages layout optimization and spatial arrangement."""

    def __init__(self):
        """Initialize the layout manager."""
        self.min_font_size = 8
        self.max_font_size = 16
        self.default_font_size = 10
        logger.info("LayoutManager initialized")

    def create_layout(
        self,
        image: np.ndarray,
        translated_blocks: List[Dict]
    ) -> LayoutConfiguration:
        """Create optimized layout configuration for bilingual PDF.

        Args:
            image: Original processed image.
            translated_blocks: List of translated text blocks with bounding boxes.

        Returns:
            LayoutConfiguration object with spatial arrangement.
        """
        height, width = image.shape[:2]
        logger.info(f"Creating layout for image: {width}x{height}")

        # Convert to TextBlock objects
        text_blocks = self._create_text_blocks(translated_blocks)

        # Sort blocks by reading order (top to bottom, left to right)
        text_blocks = self._sort_blocks_by_reading_order(text_blocks)

        # Estimate font sizes based on original dimensions
        text_blocks = self._estimate_font_sizes(text_blocks, height)

        # Create layout configuration
        config = LayoutConfiguration(
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
            layout_mode='side-by-side'
        )

        # Optimize layout dimensions
        config = self._optimize_layout_dimensions(config)

        logger.info(
            f"Layout created: {len(text_blocks)} blocks, "
            f"{config.layout_mode} mode"
        )

        return config

    def _create_text_blocks(
        self,
        translated_blocks: List[Dict]
    ) -> List[TextBlock]:
        """Convert raw block data to TextBlock objects.

        Args:
            translated_blocks: List of dictionaries with translation data.

        Returns:
            List of TextBlock objects.
        """
        text_blocks = []

        for block in translated_blocks:
            bbox = block['bbox']
            text_blocks.append(TextBlock(
                original_text=block['original'],
                translated_text=block['translated'],
                x=bbox['x'],
                y=bbox['y'],
                width=bbox['w'],
                height=bbox['h'],
                confidence=block['confidence']
            ))

        return text_blocks

    def _sort_blocks_by_reading_order(
        self,
        blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Sort text blocks in natural reading order.

        Args:
            blocks: Unsorted list of text blocks.

        Returns:
            Sorted list of text blocks.
        """
        # Group blocks by approximate rows (using y-coordinate)
        if not blocks:
            return blocks

        # Sort by y-coordinate first
        blocks_sorted = sorted(blocks, key=lambda b: b.y)

        # Group into rows (blocks with similar y-coordinates)
        rows = []
        current_row = [blocks_sorted[0]]
        row_threshold = blocks_sorted[0].height * 0.5

        for block in blocks_sorted[1:]:
            # Check if block belongs to current row
            if abs(block.y - current_row[0].y) <= row_threshold:
                current_row.append(block)
            else:
                # Sort current row by x-coordinate
                current_row.sort(key=lambda b: b.x)
                rows.append(current_row)
                current_row = [block]

        # Add last row
        if current_row:
            current_row.sort(key=lambda b: b.x)
            rows.append(current_row)

        # Flatten rows back to single list
        sorted_blocks = [block for row in rows for block in row]

        logger.info(f"Sorted {len(sorted_blocks)} blocks into {len(rows)} rows")
        return sorted_blocks

    def _estimate_font_sizes(
        self,
        blocks: List[TextBlock],
        image_height: int
    ) -> List[TextBlock]:
        """Estimate appropriate font sizes based on original text dimensions.

        Args:
            blocks: List of text blocks.
            image_height: Original image height.

        Returns:
            List of text blocks with estimated font sizes.
        """
        for block in blocks:
            # Estimate font size from bounding box height
            # Typical ratio: font size ≈ 0.7 * text height
            estimated_size = int(block.height * 0.7)

            # Clamp to reasonable range
            block.font_size = max(
                self.min_font_size,
                min(estimated_size, self.max_font_size)
            )

            # Adjust line spacing based on font size
            if block.font_size <= 10:
                block.line_spacing = 1.3
            else:
                block.line_spacing = 1.2

        return blocks

    def _optimize_layout_dimensions(
        self,
        config: LayoutConfiguration
    ) -> LayoutConfiguration:
        """Optimize page dimensions to fit content.

        Args:
            config: Initial layout configuration.

        Returns:
            Optimized layout configuration.
        """
        # Calculate scale factor to fit image on half the page
        image_half_width = (
            config.page_width - 
            config.margins['left'] - 
            config.margins['middle'] / 2
        ) / 2

        scale_factor = image_half_width / config.image_width

        # Adjust if image is too tall
        max_image_height = (
            config.page_height - 
            config.margins['top'] - 
            config.margins['bottom']
        )
        scaled_height = config.image_height * scale_factor

        if scaled_height > max_image_height:
            scale_factor = max_image_height / config.image_height

        logger.info(f"Image scale factor: {scale_factor:.3f}")

        return config

    def calculate_text_position(
        self,
        block: TextBlock,
        config: LayoutConfiguration,
        scale_factor: float
    ) -> Tuple[int, int]:
        """Calculate position for translated text on PDF.

        Args:
            block: Text block to position.
            config: Layout configuration.
            scale_factor: Image scaling factor.

        Returns:
            Tuple of (x, y) coordinates in PDF space.
        """
        # Calculate position relative to the translation section
        # Original image is on left, translation on right

        # Right side starts after image + middle margin
        translation_start_x = (
            config.margins['left'] +
            config.image_width * scale_factor +
            config.margins['middle']
        )

        # Map original text position to translation position
        x = translation_start_x + (block.x * scale_factor)
        y = config.margins['top'] + (block.y * scale_factor)

        return int(x), int(y)

    def estimate_text_height(
        self,
        text: str,
        font_size: int,
        max_width: int,
        chars_per_line: Optional[int] = None
    ) -> int:
        """Estimate the height needed for text rendering.

        Args:
            text: Text to render.
            font_size: Font size in points.
            max_width: Maximum width available.
            chars_per_line: Estimated characters per line.

        Returns:
            Estimated height in points.
        """
        if not text:
            return font_size

        # Rough estimate: average character width ≈ 0.5 * font_size
        if chars_per_line is None:
            char_width = font_size * 0.5
            chars_per_line = int(max_width / char_width)

        # Calculate number of lines needed
        num_lines = max(1, len(text) // chars_per_line + 1)

        # Total height with line spacing
        line_height = font_size * 1.2  # 1.2 is typical line spacing
        total_height = num_lines * line_height

        return int(total_height)

    def detect_columns(
        self,
        blocks: List[TextBlock],
        threshold: int = 50
    ) -> List[List[TextBlock]]:
        """Detect if text is organized in columns.

        Args:
            blocks: List of text blocks.
            threshold: Minimum horizontal gap to consider a column break.

        Returns:
            List of columns, each containing text blocks.
        """
        if not blocks:
            return []

        # Sort blocks by x-coordinate
        sorted_blocks = sorted(blocks, key=lambda b: b.x)

        columns = [[sorted_blocks[0]]]

        for block in sorted_blocks[1:]:
            last_column = columns[-1]
            last_block = max(last_column, key=lambda b: b.right)

            # Check if block starts significantly after the previous column
            if block.x - last_block.right > threshold:
                # Start new column
                columns.append([block])
            else:
                # Add to current column
                last_column.append(block)

        logger.info(f"Detected {len(columns)} column(s)")
        return columns
