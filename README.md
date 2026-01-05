# Book Translator

A command-line tool to translate book pages from images to bilingual PDFs using OCR and machine translation.

## Features

- Extract text from book page images using OCR
- Translate text to your target language
- Generate bilingual PDFs with original and translated text side-by-side
- Support for multiple languages
- Configurable DPI for OCR processing

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (install separately)

### Install from source

```bash
git clone https://github.com/jarvasquezromo/book_translator.git
cd book_translator
pip install -e .
```

### Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:** Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

After installation:

```bash
book-translator input_image.jpg output.pdf --target-lang es
```

### Command-line options

- `input`: Path to input image file
- `output`: Path to output PDF file
- `--source-lang`: Source language code (default: auto-detect)
- `--target-lang`: Target language code (default: en)
- `--dpi`: DPI for OCR processing (default: 300)

### Examples

Translate a Spanish book page to English:
```bash
book-translator page.jpg translated.pdf --source-lang es --target-lang en
```

Auto-detect source language and translate to French:
```bash
book-translator page.jpg translated.pdf --target-lang fr
```

## Development

### Setup

```bash
git clone https://github.com/jarvasquezromo/book_translator.git
cd book_translator
pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

## License

Apache 2.0 License
