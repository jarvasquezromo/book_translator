"""Configuration management module.

This module provides centralized configuration for the book translator,
making it easy to adjust settings without modifying core logic.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    """Image processing configuration."""
    target_width: int = 1200
    min_contour_area: int = 10000
    edge_detection_threshold1: int = 50
    edge_detection_threshold2: int = 150
    gaussian_blur_kernel: tuple = (5, 5)
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_constant: int = 2
    denoise_strength: int = 10


@dataclass
class OCRConfig:
    """OCR processing configuration."""
    default_dpi: int = 300
    min_confidence: int = 30
    tesseract_oem: int = 3  # OCR Engine Mode: 3 = Default
    tesseract_psm: int = 6  # Page Segmentation Mode: 6 = Uniform block
    languages: str = 'eng'  # Default language, can be 'eng+fra' for multiple


@dataclass
class TranslationConfig:
    """Translation service configuration."""
    default_source_lang: str = 'auto'
    default_target_lang: str = 'en'
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_delay: float = 0.5
    max_text_length: int = 5000


@dataclass
class LayoutConfig:
    """PDF layout configuration."""
    page_width: int = 595  # A4 width in points
    page_height: int = 842  # A4 height in points
    margin_top: int = 50
    margin_bottom: int = 50
    margin_left: int = 50
    margin_right: int = 50
    margin_middle: int = 30
    min_font_size: int = 8
    max_font_size: int = 16
    default_font_size: int = 10
    line_spacing: float = 1.2


@dataclass
class PDFConfig:
    """PDF generation configuration."""
    default_font: str = 'Helvetica'
    title: str = 'Book Translation'
    author: str = 'Book Translator'
    subject: str = 'Bilingual Document'
    include_page_numbers: bool = False
    include_separator_line: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    image: ImageConfig = None
    ocr: OCRConfig = None
    translation: TranslationConfig = None
    layout: LayoutConfig = None
    pdf: PDFConfig = None
    log_level: str = 'INFO'
    log_file: str = 'book_translator.log'

    def __post_init__(self):
        """Initialize nested configs with defaults if not provided."""
        if self.image is None:
            self.image = ImageConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.translation is None:
            self.translation = TranslationConfig()
        if self.layout is None:
            self.layout = LayoutConfig()
        if self.pdf is None:
            self.pdf = PDFConfig()


class ConfigManager:
    """Manages application configuration with save/load functionality."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file (optional).
        """
        self.config_path = config_path or Path('config.json')
        self.config = self._load_config()
        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default.

        Returns:
            AppConfig instance.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Reconstruct nested dataclasses
                config = AppConfig(
                    image=ImageConfig(**data.get('image', {})),
                    ocr=OCRConfig(**data.get('ocr', {})),
                    translation=TranslationConfig(**data.get('translation', {})),
                    layout=LayoutConfig(**data.get('layout', {})),
                    pdf=PDFConfig(**data.get('pdf', {})),
                    log_level=data.get('log_level', 'INFO'),
                    log_file=data.get('log_file', 'book_translator.log')
                )

                logger.info(f"Configuration loaded from {self.config_path}")
                return config

            except Exception as e:
                logger.warning(
                    f"Failed to load config from {self.config_path}: {e}. "
                    f"Using defaults."
                )
                return AppConfig()
        else:
            logger.info("No config file found, using defaults")
            return AppConfig()

    def save_config(self) -> bool:
        """Save current configuration to file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Convert to dictionary
            config_dict = {
                'image': asdict(self.config.image),
                'ocr': asdict(self.config.ocr),
                'translation': asdict(self.config.translation),
                'layout': asdict(self.config.layout),
                'pdf': asdict(self.config.pdf),
                'log_level': self.config.log_level,
                'log_file': self.config.log_file
            }

            # Save to file with pretty formatting
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def get_image_config(self) -> ImageConfig:
        """Get image processing configuration."""
        return self.config.image

    def get_ocr_config(self) -> OCRConfig:
        """Get OCR configuration."""
        return self.config.ocr

    def get_translation_config(self) -> TranslationConfig:
        """Get translation configuration."""
        return self.config.translation

    def get_layout_config(self) -> LayoutConfig:
        """Get layout configuration."""
        return self.config.layout

    def get_pdf_config(self) -> PDFConfig:
        """Get PDF generation configuration."""
        return self.config.pdf

    def update_config(self, section: str, **kwargs) -> None:
        """Update configuration section.

        Args:
            section: Config section name ('image', 'ocr', etc.).
            **kwargs: Configuration parameters to update.

        Raises:
            ValueError: If section is invalid.
        """
        section_map = {
            'image': self.config.image,
            'ocr': self.config.ocr,
            'translation': self.config.translation,
            'layout': self.config.layout,
            'pdf': self.config.pdf
        }

        if section not in section_map:
            raise ValueError(f"Invalid config section: {section}")

        config_obj = section_map[section]

        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                logger.info(f"Updated {section}.{key} = {value}")
            else:
                logger.warning(
                    f"Unknown config parameter: {section}.{key}"
                )

    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        self.config = AppConfig()
        logger.info("Configuration reset to defaults")

    def create_example_config(self) -> None:
        """Create an example configuration file with comments."""
        example_path = Path('config.example.json')

        example_config = {
            "_comment": "Book Translator Configuration File",
            "_instructions": "Copy to config.json and modify as needed",
            "image": {
                "_description": "Image processing settings",
                "target_width": 1200,
                "min_contour_area": 10000,
                "edge_detection_threshold1": 50,
                "edge_detection_threshold2": 150,
                "gaussian_blur_kernel": [5, 5],
                "adaptive_threshold_block_size": 11,
                "adaptive_threshold_constant": 2,
                "denoise_strength": 10
            },
            "ocr": {
                "_description": "OCR settings",
                "default_dpi": 300,
                "min_confidence": 30,
                "tesseract_oem": 3,
                "tesseract_psm": 6,
                "languages": "eng"
            },
            "translation": {
                "_description": "Translation API settings",
                "default_source_lang": "auto",
                "default_target_lang": "en",
                "max_retries": 3,
                "retry_delay": 1.0,
                "batch_delay": 0.5,
                "max_text_length": 5000
            },
            "layout": {
                "_description": "PDF layout settings",
                "page_width": 595,
                "page_height": 842,
                "margin_top": 50,
                "margin_bottom": 50,
                "margin_left": 50,
                "margin_right": 50,
                "margin_middle": 30,
                "min_font_size": 8,
                "max_font_size": 16,
                "default_font_size": 10,
                "line_spacing": 1.2
            },
            "pdf": {
                "_description": "PDF generation settings",
                "default_font": "Helvetica",
                "title": "Book Translation",
                "author": "Book Translator",
                "subject": "Bilingual Document",
                "include_page_numbers": False,
                "include_separator_line": True
            },
            "log_level": "INFO",
            "log_file": "book_translator.log"
        }

        try:
            with open(example_path, 'w', encoding='utf-8') as f:
                json.dump(example_config, f, indent=2, ensure_ascii=False)

            logger.info(f"Example configuration created: {example_path}")
        except Exception as e:
            logger.error(f"Failed to create example config: {e}")


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get or create the global configuration manager.

    Args:
        config_path: Path to configuration file (optional).

    Returns:
        ConfigManager instance.
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_path)

    return _config_manager
