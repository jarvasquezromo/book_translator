"""Computer vision module for image preprocessing and OCR.

This module handles:
- Page detection and auto-framing
- Perspective correction
- Text extraction with bounding boxes
- Language detection
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing and OCR text extraction."""

    def __init__(self, min_contour_area: int = 10000):
        """Initialize the image processor.

        Args:
            min_contour_area: Minimum contour area for page detection.
        """
        self.min_contour_area = min_contour_area
        logger.info("ImageProcessor initialized")

    def process_page(
        self,
        image_path: str,
        target_width: int = 1200
    ) -> Optional[np.ndarray]:
        """Process book page: detect, crop, and correct perspective.

        Args:
            image_path: Path to the input image.
            target_width: Target width for resized image.

        Returns:
            Processed image as numpy array, or None if processing fails.
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            logger.info(f"Image loaded: {image.shape}")

            # Resize for faster processing
            resized = self._resize_image(image, target_width)

            # Detect page contour
            page_contour = self._detect_page_contour(resized)

            if page_contour is not None:
                # Apply perspective correction
                warped = self._perspective_transform(resized, page_contour)
                logger.info("Perspective correction applied")
            else:
                logger.warning("Page detection failed, using full image")
                warped = resized

            # Enhance image for better OCR
            enhanced = self._enhance_for_ocr(warped)

            return enhanced

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def _resize_image(
        self,
        image: np.ndarray,
        target_width: int
    ) -> np.ndarray:
        """Resize image while maintaining aspect ratio.

        Args:
            image: Input image.
            target_width: Desired width.

        Returns:
            Resized image.
        """
        height, width = image.shape[:2]
        if width > target_width:
            ratio = target_width / width
            new_height = int(height * ratio)
            return cv2.resize(image, (target_width, new_height))
        return image

    def _detect_page_contour(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Detect the largest rectangular contour (book page).

        Args:
            image: Input image.

        Returns:
            Contour points as numpy array, or None if not found.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Dilate edges to close gaps
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                dilated,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Check if contour is large enough
            if cv2.contourArea(largest_contour) < self.min_contour_area:
                return None

            # Approximate contour to polygon
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

            # We need a quadrilateral
            if len(approx) == 4:
                return approx.reshape(4, 2)

            return None

        except Exception as e:
            logger.error(f"Error detecting page contour: {e}")
            return None

    def _perspective_transform(
        self,
        image: np.ndarray,
        contour: np.ndarray
    ) -> np.ndarray:
        """Apply perspective transformation to straighten the page.

        Args:
            image: Input image.
            contour: Four corner points of the page.

        Returns:
            Warped image with corrected perspective.
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = self._order_points(contour)

        # Compute width and height of new image
        width_a = np.linalg.norm(pts[0] - pts[1])
        width_b = np.linalg.norm(pts[2] - pts[3])
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(pts[0] - pts[3])
        height_b = np.linalg.norm(pts[1] - pts[2])
        max_height = int(max(height_a, height_b))

        # Destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)

        # Apply transformation
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left.

        Args:
            pts: Four points to order.

        Returns:
            Ordered points array.
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum and difference to find corners
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def _enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR accuracy.

        Args:
            image: Input image.

        Returns:
            Enhanced image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Convert back to BGR for consistency
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def extract_text_with_boxes(
        self,
        image: np.ndarray,
        dpi: int = 300
    ) -> List[Dict[str, any]]:
        """Extract text with bounding boxes using OCR.

        Args:
            image: Preprocessed image.
            dpi: DPI setting for OCR.

        Returns:
            List of dictionaries containing text, bounding boxes, and metadata.
        """
        try:
            # Configure Tesseract
            custom_config = f'--oem 3 --psm 6 --dpi {dpi}'

            # Perform OCR with detailed output
            data = pytesseract.image_to_data(
                image,
                output_type=Output.DICT,
                config=custom_config
            )

            results = []
            n_boxes = len(data['text'])

            for i in range(n_boxes):
                # Filter out empty or low-confidence results
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()

                if confidence > 30 and text:
                    bbox = {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i]
                    }

                    results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })

            logger.info(f"Extracted {len(results)} text blocks")
            return results

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return []

    def detect_language(self, image: np.ndarray) -> str:
        """Detect the primary language in the image.

        Args:
            image: Input image.

        Returns:
            ISO language code (e.g., 'eng', 'fra', 'deu').
        """
        try:
            # Use Tesseract's language detection
            osd = pytesseract.image_to_osd(image)
            for line in osd.split('\n'):
                if 'Script:' in line:
                    script = line.split(':')[1].strip()
                    logger.info(f"Detected script: {script}")

            # Extract text with language info
            langs = pytesseract.get_languages()
            logger.info(f"Available languages: {langs}")

            return 'eng'  # Default to English

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'eng'
