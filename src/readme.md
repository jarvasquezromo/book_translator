# Book Translation Application

A professional, cross-platform Python application for automated book translation from images. This tool processes book page images, extracts text using OCR, translates it, and generates a bilingual PDF with the original image and translated text side-by-side.

## Features

- **Intelligent Image Processing**: Automatic page detection, cropping, and perspective correction
- **Advanced OCR**: Text extraction with bounding boxes and confidence scores
- **Multi-language Support**: Automatic language detection and translation to 100+ languages
- **Professional PDF Output**: Bilingual layout preserving original text positioning
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Robust Error Handling**: Comprehensive logging and exception management

## Architecture

```
book-translator/
├── main.py          # Application orchestrator and CLI
├── vision.py        # Image processing and OCR
├── translator.py    # Translation engine
├── layout.py        # Layout management
├── pdf_gen.py       # PDF generation
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

### Module Interactions

1. **main.py** → Orchestrates the entire workflow
2. **vision.py** → Processes image, detects page, performs OCR
3. **translator.py** → Translates extracted text blocks
4. **layout.py** → Analyzes spatial layout and optimizes positioning
5. **pdf_gen.py** → Generates bilingual PDF with ReportLab

## Installation

### Prerequisites

1. **Python 3.8+** (tested with Python 3.9-3.11)

2. **Tesseract-OCR** (required for pytesseract):
   
   **Windows:**
   ```bash
   # Download and install from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   
   # Add Tesseract to PATH or set in code:
   # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr
   sudo apt-get install libtesseract-dev
   ```

### Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py input_image.jpg output.pdf
```

### Advanced Options

```bash
python main.py input_image.jpg output.pdf \
    --source-lang auto \
    --target-lang es \
    --dpi 300
```

### Command-Line Arguments

- `input`: Path to input image file (required)
- `output`: Path to output PDF file (required)
- `--source-lang`: Source language code (default: `auto` for auto-detection)
- `--target-lang`: Target language code (default: `en`)
- `--dpi`: DPI for OCR processing (default: `300`)

### Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh-CN` - Chinese (Simplified)
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic

For full list, see: [Google Translate Language Codes](https://py-googletrans.readthedocs.io/en/latest/#googletrans-languages)

## Examples

### Example 1: English to Spanish

```bash
python main.py book_page.jpg translated_page.pdf --target-lang es
```

### Example 2: Auto-detect source, translate to French

```bash
python main.py scan.png output.pdf --source-lang auto --target-lang fr
```

### Example 3: High-DPI processing

```bash
python main.py photo.jpg result.pdf --dpi 600
```

## Configuration

### Adjusting Image Processing

Edit `vision.py` to customize:
- `min_contour_area`: Minimum page detection area (default: 10000)
- `target_width`: Image resize width (default: 1200)

### Customizing Layout

Edit `layout.py` to adjust:
- `margins`: Page margins dictionary
- `min_font_size`, `max_font_size`: Font size range
- `layout_mode`: 'side-by-side' or 'overlay'

### PDF Settings

Edit `pdf_gen.py` to configure:
- `page_size`: Change from A4 to other sizes
- `default_font`: Font family
- Text wrapping behavior

## Troubleshooting

### Tesseract Not Found

**Error:** `pytesseract.pytesseract.TesseractNotFoundError`

**Solution:**
```python
# Add to vision.py after imports:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'
```

### Poor OCR Accuracy

**Solutions:**
1. Increase DPI: `--dpi 600`
2. Ensure good image quality (high resolution, good lighting)
3. Use perspective correction (automatic in the app)
4. Install additional Tesseract language packs:
   ```bash
   # Ubuntu example for Spanish
   sudo apt-get install tesseract-ocr-spa
   ```

### Translation Errors

**Error:** `TranslationNotFound` or rate limiting

**Solutions:**
1. Check internet connection
2. Add delays between requests (adjust `batch_delay` in `translator.py`)
3. Consider using paid translation API for production

### Memory Issues with Large Images

**Solution:** Reduce `target_width` in `vision.py`:
```python
processed_image = self.image_processor.process_page(
    str(image_path),
    target_width=800  # Reduce from default 1200
)
```

## Best Practices

1. **Image Quality**: Use high-resolution scans (300+ DPI)
2. **Lighting**: Ensure even lighting without shadows
3. **Page Detection**: Place book on contrasting background
4. **Batch Processing**: Process multiple pages in a loop
5. **Error Handling**: Always check return values and logs

## Development

### Code Style

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):
- Type hints for all function signatures
- Docstrings for all public methods
- PEP 8 naming conventions
- Maximum line length: 79 characters

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests (when implemented)
pytest tests/ -v --cov=.
```

### Logging

All modules log to both console and `book_translator.log`:
- INFO: Normal operations
- WARNING: Potential issues
- ERROR: Failures requiring attention

Adjust logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
```

## Performance Optimization

### Speed Improvements

1. **Reduce Image Size**: Lower `target_width` in processing
2. **Adjust OCR DPI**: Lower DPI for faster processing
3. **Batch Translation**: Process multiple pages together
4. **Disable Enhancements**: Skip certain image enhancements

### Quality Improvements

1. **Increase DPI**: Higher DPI for better OCR accuracy
2. **Better Image Preprocessing**: Adjust thresholding parameters
3. **Manual Page Detection**: Skip auto-detection for clean scans

## Limitations

- **OCR Accuracy**: Depends on image quality and text clarity
- **Layout Preservation**: Complex layouts may not be perfectly preserved
- **Translation Quality**: Limited by Google Translate API
- **Rate Limiting**: Free translation API has usage limits
- **Single Page**: Currently processes one page at a time

## Future Enhancements

- [ ] Batch processing for multiple pages
- [ ] Multi-page PDF support
- [ ] Custom translation API integration
- [ ] GUI application
- [ ] Layout templates for different book types
- [ ] Support for tables and complex formatting
- [ ] Cloud deployment options

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow the Google Python Style Guide
4. Add tests for new functionality
5. Submit a pull request

## License

This project is provided as-is for educational and personal use.

## Support

For issues and questions:
1. Check the Troubleshooting section
2. Review logs in `book_translator.log`
3. Ensure all dependencies are properly installed
4. Verify Tesseract-OCR installation

## Acknowledgments

- **OpenCV**: Image processing
- **Tesseract-OCR**: Text recognition
- **Deep Translator**: Translation services
- **ReportLab**: PDF generation
