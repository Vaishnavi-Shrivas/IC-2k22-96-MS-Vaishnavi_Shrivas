"""
image_enhancement.py
----------------------
Cluster: Image Enhancement

Applies basic image enhancement operations using OpenCV to improve the
visual quality of a photo: better contrast, corrected brightness, reduced
noise, sharper detail, and upscaled resolution.

Supported formats: JPG, JPEG, PNG, BMP, WEBP, TIFF

Usage:
    python image_enhancement.py <path-to-image>
"""

import os
import sys

import cv2
import numpy as np

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _load_image(input_path):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Unable to load image. It may be corrupt or unsupported.")
    return image


def enhance_contrast(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the
    lightness channel in LAB color space. Boosts local contrast without
    blowing out colors the way a naive global histogram equalization would.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def correct_brightness(image, target_mean=130.0):
    """
    Automatic gamma correction: measures the image's current average
    brightness and nudges it toward a mid-range target, so dark/underexposed
    photos get lifted without manual tuning.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_mean = max(1.0, float(np.mean(gray)))
    # Solve for the gamma that maps current_mean -> target_mean:
    # (current_mean/255)^gamma = target_mean/255
    gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
    gamma = np.clip(gamma, 0.3, 3.0)  # avoid extreme corrections on edge-case images

    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def denoise(image):
    """Non-local means denoising — removes sensor/compression noise while keeping edges."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)


def sharpen(image):
    """Unsharp mask: blur the image, then push the original away from that blur."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def upscale(image, scale=2):
    """
    Upscale resolution using high-quality cubic interpolation.
    (A learned super-resolution model would do better, but needs extra
    pretrained weight files — cubic interpolation needs nothing extra
    and still gives a clean, sharper-looking upscale.)
    """
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def enhance_image(input_path, output_dir):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    image = _load_image(input_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Contrast enhancement
    contrast_enhanced = enhance_contrast(image)
    cv2.imwrite(os.path.join(output_dir, "contrast_enhanced.jpg"), contrast_enhanced)

    # 2. Brightness correction (auto gamma)
    brightness_corrected = correct_brightness(image)
    cv2.imwrite(os.path.join(output_dir, "brightness_corrected.jpg"), brightness_corrected)

    # 3. Denoising
    denoised = denoise(image)
    cv2.imwrite(os.path.join(output_dir, "denoised.jpg"), denoised)

    # 4. Sharpening
    sharpened = sharpen(image)
    cv2.imwrite(os.path.join(output_dir, "sharpened.jpg"), sharpened)

    # 5. Upscaling (2x)
    upscaled = upscale(image, scale=2)
    cv2.imwrite(os.path.join(output_dir, "upscaled_2x.jpg"), upscaled)

    # 6. Fully enhanced: all fixes combined in one pipeline
    combined = denoise(image)
    combined = correct_brightness(combined)
    combined = enhance_contrast(combined)
    combined = sharpen(combined)
    cv2.imwrite(os.path.join(output_dir, "fully_enhanced.jpg"), combined)

    print("\nImage enhancement completed successfully.")
    print("\nGenerated outputs:")
    for name in ("contrast_enhanced.jpg", "brightness_corrected.jpg", "denoised.jpg",
                 "sharpened.jpg", "upscaled_2x.jpg", "fully_enhanced.jpg"):
        print(f" - {name}")


def main():
    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python image_enhancement.py <path-to-image>")
        return

    input_path = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), "output")

    try:
        enhance_image(input_path, output_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError during image processing: {e}")


if __name__ == "__main__":
    main()