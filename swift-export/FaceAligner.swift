import UIKit
import CoreImage
import Accelerate

/// Service for aligning and cropping detected faces to the format required by MobileFaceNet.
///
/// The model expects:
/// - Size: 112x112 pixels
/// - Format: RGB
/// - Normalization: [-1, 1] (pixel values mapped from [0, 255])
///
/// This class handles:
/// 1. Cropping the face region from the original image
/// 2. Applying a margin around the face for context
/// 3. Resizing to 112x112
/// 4. Converting to the correct pixel format
///
/// For best results, the face should be roughly centered and upright.
/// Advanced alignment using landmarks (rotating to make eyes horizontal)
/// can improve accuracy but is not strictly required.
class FaceAligner {

    // MARK: - Configuration

    /// Output size expected by MobileFaceNet
    static let outputSize = CGSize(width: 112, height: 112)

    /// Margin to add around the detected face bounding box (as fraction of face size)
    /// This provides context around the face which helps the model
    private let marginFraction: CGFloat = 0.3

    // MARK: - Public API

    /// Crop and align a face from an image
    /// - Parameters:
    ///   - image: Source image containing the face
    ///   - face: Detected face with bounding box (and optional landmarks)
    /// - Returns: 112x112 aligned face image, or nil if cropping fails
    func alignFace(from image: UIImage, face: DetectedFace) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        // Convert Vision coordinates (bottom-left origin, normalized) to UIKit coordinates
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        let faceRect = convertToImageCoordinates(face.boundingBox, imageSize: imageSize)

        // Add margin around face
        let expandedRect = expandRect(faceRect, by: marginFraction, within: imageSize)

        // Crop the face region
        guard let croppedCGImage = cgImage.cropping(to: expandedRect) else {
            return nil
        }

        // Resize to 112x112
        let resizedImage = resizeImage(UIImage(cgImage: croppedCGImage), to: Self.outputSize)

        return resizedImage
    }

    /// Align face using 5-point landmarks for better accuracy
    /// This applies an affine transform to:
    /// 1. Rotate the face so eyes are horizontal
    /// 2. Scale to standard size
    /// 3. Translate to center the face
    ///
    /// - Parameters:
    ///   - image: Source image
    ///   - face: Detected face with landmarks
    /// - Returns: Aligned 112x112 face image
    func alignFaceWithLandmarks(from image: UIImage, face: DetectedFace) -> UIImage? {
        guard let landmarks = face.landmarks,
              let cgImage = image.cgImage else {
            // Fall back to simple alignment if no landmarks
            return alignFace(from: image, face: face)
        }

        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)

        // Standard destination points for ArcFace/InsightFace alignment (112x112)
        // These are the expected positions of facial landmarks in the output image
        let dstPoints: [CGPoint] = [
            CGPoint(x: 38.2946, y: 51.6963),   // Left eye
            CGPoint(x: 73.5318, y: 51.5014),   // Right eye
            CGPoint(x: 56.0252, y: 71.7366),   // Nose
            CGPoint(x: 41.5493, y: 92.3655),   // Left mouth
            CGPoint(x: 70.7299, y: 92.2041)    // Right mouth
        ]

        // Convert landmarks from Vision coordinates to image coordinates
        let srcPoints: [CGPoint] = [
            convertPointToImage(landmarks.leftEye, imageSize: imageSize),
            convertPointToImage(landmarks.rightEye, imageSize: imageSize),
            convertPointToImage(landmarks.nose, imageSize: imageSize),
            convertPointToImage(landmarks.leftMouth, imageSize: imageSize),
            convertPointToImage(landmarks.rightMouth, imageSize: imageSize)
        ]

        // Compute similarity transform from source to destination points
        guard let transform = computeSimilarityTransform(from: srcPoints, to: dstPoints) else {
            return alignFace(from: image, face: face)
        }

        // Apply transform to get aligned face
        let alignedImage = applyTransform(transform, to: image, outputSize: Self.outputSize)

        return alignedImage
    }

    // MARK: - Pixel Buffer Conversion

    /// Convert aligned face image to CVPixelBuffer for CoreML
    /// The model expects:
    /// - Pixel format: 32BGRA or 32ARGB
    /// - Size: 112x112
    ///
    /// Note: Normalization to [-1, 1] is typically done by the CoreML model's
    /// preprocessing configuration, not here.
    func createPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        // Create pixel buffer
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        // Draw image into pixel buffer
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }

    // MARK: - Private Helpers

    /// Convert Vision normalized coordinates to image pixel coordinates
    /// Vision uses bottom-left origin with y pointing up
    /// UIKit uses top-left origin with y pointing down
    private func convertToImageCoordinates(_ normalizedRect: CGRect, imageSize: CGSize) -> CGRect {
        return CGRect(
            x: normalizedRect.origin.x * imageSize.width,
            y: (1 - normalizedRect.origin.y - normalizedRect.height) * imageSize.height,
            width: normalizedRect.width * imageSize.width,
            height: normalizedRect.height * imageSize.height
        )
    }

    /// Convert normalized point to image coordinates
    private func convertPointToImage(_ point: CGPoint, imageSize: CGSize) -> CGPoint {
        return CGPoint(
            x: point.x * imageSize.width,
            y: (1 - point.y) * imageSize.height
        )
    }

    /// Expand a rectangle by a fraction of its size, staying within image bounds
    private func expandRect(_ rect: CGRect, by fraction: CGFloat, within bounds: CGSize) -> CGRect {
        let expandX = rect.width * fraction
        let expandY = rect.height * fraction

        var expanded = rect.insetBy(dx: -expandX, dy: -expandY)

        // Clamp to image bounds
        expanded.origin.x = max(0, expanded.origin.x)
        expanded.origin.y = max(0, expanded.origin.y)
        expanded.size.width = min(expanded.size.width, bounds.width - expanded.origin.x)
        expanded.size.height = min(expanded.size.height, bounds.height - expanded.origin.y)

        return expanded
    }

    /// Resize image to target size
    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }

        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    /// Compute similarity transform (rotation, scale, translation) between two point sets
    /// Uses least squares fitting
    private func computeSimilarityTransform(from src: [CGPoint], to dst: [CGPoint]) -> CGAffineTransform? {
        guard src.count == dst.count, src.count >= 2 else { return nil }

        // Compute centroids
        var srcCentroid = CGPoint.zero
        var dstCentroid = CGPoint.zero
        for i in 0..<src.count {
            srcCentroid.x += src[i].x
            srcCentroid.y += src[i].y
            dstCentroid.x += dst[i].x
            dstCentroid.y += dst[i].y
        }
        srcCentroid.x /= CGFloat(src.count)
        srcCentroid.y /= CGFloat(src.count)
        dstCentroid.x /= CGFloat(dst.count)
        dstCentroid.y /= CGFloat(dst.count)

        // Center the points
        var srcCentered = src.map { CGPoint(x: $0.x - srcCentroid.x, y: $0.y - srcCentroid.y) }
        var dstCentered = dst.map { CGPoint(x: $0.x - dstCentroid.x, y: $0.y - dstCentroid.y) }

        // Compute scale
        var srcNorm: CGFloat = 0
        var dstNorm: CGFloat = 0
        for i in 0..<src.count {
            srcNorm += srcCentered[i].x * srcCentered[i].x + srcCentered[i].y * srcCentered[i].y
            dstNorm += dstCentered[i].x * dstCentered[i].x + dstCentered[i].y * dstCentered[i].y
        }
        let scale = sqrt(dstNorm / srcNorm)

        // Compute rotation using SVD approximation
        var sumXX: CGFloat = 0, sumXY: CGFloat = 0
        var sumYX: CGFloat = 0, sumYY: CGFloat = 0
        for i in 0..<src.count {
            sumXX += srcCentered[i].x * dstCentered[i].x
            sumXY += srcCentered[i].x * dstCentered[i].y
            sumYX += srcCentered[i].y * dstCentered[i].x
            sumYY += srcCentered[i].y * dstCentered[i].y
        }

        let angle = atan2(sumXY - sumYX, sumXX + sumYY)

        // Build transform: translate to origin, scale, rotate, translate to destination
        var transform = CGAffineTransform.identity
        transform = transform.translatedBy(x: dstCentroid.x, y: dstCentroid.y)
        transform = transform.rotated(by: angle)
        transform = transform.scaledBy(x: scale, y: scale)
        transform = transform.translatedBy(x: -srcCentroid.x, y: -srcCentroid.y)

        return transform
    }

    /// Apply affine transform to image
    private func applyTransform(_ transform: CGAffineTransform, to image: UIImage, outputSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(outputSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }

        guard let context = UIGraphicsGetCurrentContext() else { return nil }

        // Apply transform and draw
        context.concatenate(transform)
        image.draw(at: .zero)

        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
