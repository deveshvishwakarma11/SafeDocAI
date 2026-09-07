"""
SafeDocAI - Utility Functions

Phase 1:
- File validation
- Image loading
- Image resizing
- Grayscale conversion
- Gaussian blur
- Adaptive thresholding
- Automatic deskewing
- OCR preprocessing

All processing is performed locally.
"""

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Supported File Types
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

SUPPORTED_PDF_EXTENSIONS = {
    ".pdf",
}


# ============================================================
# File Handling
# ============================================================

def validate_file(file_path: str | Path) -> Path:
    """
    Validate that the supplied path exists and is supported.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    supported_extensions = (
        SUPPORTED_IMAGE_EXTENSIONS
        | SUPPORTED_PDF_EXTENSIONS
    )

    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: "
            f"{', '.join(sorted(supported_extensions))}"
        )

    return path


def is_image_file(
    file_path: str | Path,
) -> bool:
    """Return True if the file is a supported image."""

    return (
        Path(file_path).suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    )


def is_pdf_file(
    file_path: str | Path,
) -> bool:
    """Return True if the file is a PDF."""

    return (
        Path(file_path).suffix.lower()
        in SUPPORTED_PDF_EXTENSIONS
    )


# ============================================================
# Image Loading
# ============================================================

def load_image(
    file_path: str | Path,
) -> np.ndarray:
    """
    Load an image using OpenCV.
    """

    path = validate_file(file_path)

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {path}"
        )

    return image


# ============================================================
# Image Resizing
# ============================================================

def resize_image(
    image: np.ndarray,
    max_dimension: int = 2500,
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.

    Images smaller than max_dimension are not resized.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    height, width = image.shape[:2]

    largest_dimension = max(
        height,
        width,
    )

    if largest_dimension <= max_dimension:
        return image

    scale = (
        max_dimension
        / largest_dimension
    )

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# Grayscale
# ============================================================

def to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


# ============================================================
# Gaussian Blur
# ============================================================

def apply_gaussian_blur(
    gray_image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce image noise.
    """

    if gray_image is None:
        raise ValueError(
            "Image cannot be None."
        )

    return cv2.GaussianBlur(
        gray_image,
        kernel_size,
        0,
    )


# ============================================================
# Skew Detection
# ============================================================

def estimate_skew_angle(
    gray_image: np.ndarray,
) -> float:
    """
    Estimate document/text skew angle.

    Uses:
        Canny Edge Detection
        Hough Line Transform

    The median of detected near-horizontal line angles
    is used for robustness.

    Returns:
        Estimated skew angle in degrees.
    """

    if gray_image is None:
        raise ValueError(
            "Image cannot be None."
        )

    # Detect edges.
    edges = cv2.Canny(
        gray_image,
        50,
        150,
        apertureSize=3,
    )

    height, width = gray_image.shape[:2]

    min_line_length = max(
        width // 4,
        100,
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    # No lines detected.
    if lines is None:
        return 0.0

    # ========================================================
    # IMPORTANT FIX
    #
    # OpenCV may return Hough lines in slightly different
    # shapes depending on the image/OpenCV version.
    #
    # Expected:
    #     (N, 1, 4)
    #
    # Sometimes:
    #     (N, 4)
    #
    # Normalize both formats before iterating.
    # ========================================================

    lines = np.asarray(lines)

    lines = np.squeeze(lines)

    if lines.ndim == 1:

        if lines.size != 4:
            return 0.0

        lines = lines.reshape(
            1,
            4,
        )

    if lines.ndim != 2 or lines.shape[1] != 4:
        return 0.0

    angles: list[float] = []

    for line in lines:

        # Safely unpack exactly four coordinates.
        x1, y1, x2, y2 = map(
            int,
            line,
        )

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1,
            )
        )

        # We only care about approximately horizontal
        # document/text lines.
        if -15.0 <= angle <= 15.0:

            angles.append(
                float(angle)
            )

    if not angles:
        return 0.0

    # Median is more resistant to outliers than mean.
    return float(
        np.median(angles)
    )


# ============================================================
# Image Rotation
# ============================================================

def rotate_image(
    image: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rotate an image while keeping the complete document visible.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    # No meaningful rotation required.
    if abs(angle) < 0.1:
        return image

    height, width = image.shape[:2]

    center = (
        width / 2,
        height / 2,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos = abs(
        rotation_matrix[0, 0]
    )

    sin = abs(
        rotation_matrix[0, 1]
    )

    new_width = int(
        (height * sin)
        + (width * cos)
    )

    new_height = int(
        (height * cos)
        + (width * sin)
    )

    # Adjust transformation so the entire image remains visible.
    rotation_matrix[0, 2] += (
        new_width / 2
        - center[0]
    )

    rotation_matrix[1, 2] += (
        new_height / 2
        - center[1]
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (
            new_width,
            new_height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


# ============================================================
# Auto Deskew
# ============================================================

def deskew_image(
    gray_image: np.ndarray,
) -> np.ndarray:
    """
    Automatically detect and correct document skew.
    """

    angle = estimate_skew_angle(
        gray_image
    )

    if abs(angle) < 0.1:
        return gray_image

    return rotate_image(
        gray_image,
        angle,
    )


# ============================================================
# Adaptive Thresholding
# ============================================================

def adaptive_threshold(
    gray_image: np.ndarray,
    block_size: int = 31,
    constant: int = 11,
) -> np.ndarray:
    """
    Convert grayscale image into a binary image.

    Adaptive Gaussian thresholding helps with:
    - Uneven lighting
    - Shadows
    - Scanned documents
    """

    if gray_image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if (
        block_size <= 1
        or block_size % 2 == 0
    ):
        raise ValueError(
            "block_size must be an odd number greater than 1."
        )

    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )


# ============================================================
# Binary Image Cleanup
# ============================================================

def clean_binary_image(
    binary_image: np.ndarray,
) -> np.ndarray:
    """
    Remove small noise from a binary document image.
    """

    if binary_image is None:
        raise ValueError(
            "Image cannot be None."
        )

    kernel = np.ones(
        (2, 2),
        np.uint8,
    )

    return cv2.morphologyEx(
        binary_image,
        cv2.MORPH_OPEN,
        kernel,
    )


# ============================================================
# Complete OCR Preprocessing Pipeline
# ============================================================

def preprocess_for_ocr(
    image: np.ndarray,
    max_dimension: int = 2500,
) -> np.ndarray:
    """
    Complete OCR preprocessing pipeline:

        Image
          ↓
        Resize
          ↓
        Grayscale
          ↓
        Gaussian Blur
          ↓
        Auto Deskew
          ↓
        Gaussian Blur
          ↓
        Adaptive Threshold
          ↓
        Morphological Cleanup
          ↓
        OCR-ready image
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    # 1. Resize
    image = resize_image(
        image,
        max_dimension=max_dimension,
    )

    # 2. Grayscale
    gray = to_grayscale(
        image
    )

    # 3. Initial noise reduction
    blurred = apply_gaussian_blur(
        gray
    )

    # 4. Auto deskew
    deskewed = deskew_image(
        blurred
    )

    # 5. Blur after rotation
    deskewed = apply_gaussian_blur(
        deskewed
    )

    # 6. Adaptive thresholding
    binary = adaptive_threshold(
        deskewed
    )

    # 7. Remove small noise
    cleaned = clean_binary_image(
        binary
    )

    return cleaned


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )
    print(
        " SafeDocAI - utils.py Test"
    )
    print(
        "========================================"
    )

    sample_path = (
        "data/samples/Devesh 4th sem.pdf"
    )

    print(
        f"Sample path: {sample_path}"
    )

    if is_pdf_file(sample_path):

        print(
            "PDF detected successfully."
        )

    else:

        print(
            "Sample is not detected as a PDF."
        )

    print(
        "Utility module loaded successfully."
    )