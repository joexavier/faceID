import Vision
import UIKit
import CoreImage

/// Service for detecting faces in images using Apple's Vision framework.
/// This is the first step in the face recognition pipeline.
///
/// Usage:
/// ```swift
/// let detector = FaceDetector()
/// let faces = try await detector.detectFaces(in: image)
/// for face in faces {
///     print("Face at: \(face.boundingBox)")
/// }
/// ```
class FaceDetector {

    // MARK: - Configuration

    /// Minimum confidence threshold for face detection (0-1)
    /// Higher values reduce false positives but may miss some faces
    private let minimumConfidence: Float = 0.5

    /// Whether to detect facial landmarks (eyes, nose, mouth)
    /// Landmarks are used for face alignment before embedding extraction
    private let detectLandmarks: Bool = true

    // MARK: - Public API

    /// Detect all faces in a UIImage
    /// - Parameter image: The image to scan for faces
    /// - Returns: Array of detected faces with bounding boxes and optional landmarks
    func detectFaces(in image: UIImage) async throws -> [DetectedFace] {
        guard let cgImage = image.cgImage else {
            throw FaceDetectionError.invalidImage
        }
        return try await detectFaces(in: cgImage, orientation: image.cgImageOrientation)
    }

    /// Detect all faces in a CGImage
    /// - Parameters:
    ///   - cgImage: The image to scan
    ///   - orientation: Image orientation for correct coordinate mapping
    /// - Returns: Array of detected faces
    func detectFaces(in cgImage: CGImage, orientation: CGImagePropertyOrientation = .up) async throws -> [DetectedFace] {

        // Create Vision request for face detection
        // VNDetectFaceRectanglesRequest is fast but doesn't provide landmarks
        // VNDetectFaceLandmarksRequest provides landmarks but is slower
        let request: VNImageBasedRequest

        if detectLandmarks {
            // Use landmark detection (includes face rectangles + landmark positions)
            request = VNDetectFaceLandmarksRequest()
        } else {
            // Use basic face rectangle detection (faster)
            request = VNDetectFaceRectanglesRequest()
        }

        // Create request handler with the image
        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: orientation,
            options: [:]
        )

        // Perform detection (this is synchronous, wrap in Task for async)
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])

                    // Extract results
                    let faces = self.processResults(request.results)
                    continuation.resume(returning: faces)
                } catch {
                    continuation.resume(throwing: FaceDetectionError.detectionFailed(error))
                }
            }
        }
    }

    // MARK: - Private Helpers

    /// Process Vision framework results into our DetectedFace model
    private func processResults(_ results: [Any]?) -> [DetectedFace] {
        guard let observations = results as? [VNFaceObservation] else {
            return []
        }

        return observations.compactMap { observation -> DetectedFace? in
            // Filter by confidence
            guard observation.confidence >= minimumConfidence else {
                return nil
            }

            // Extract bounding box
            // Vision uses normalized coordinates with origin at bottom-left
            // We keep them as-is; conversion to UIKit coordinates happens during cropping
            let boundingBox = observation.boundingBox

            // Extract landmarks if available
            var landmarks: FaceLandmarks? = nil
            if let visionLandmarks = observation.landmarks {
                landmarks = extractLandmarks(from: visionLandmarks, in: boundingBox)
            }

            return DetectedFace(
                boundingBox: boundingBox,
                landmarks: landmarks,
                confidence: observation.confidence
            )
        }
    }

    /// Extract landmark positions from Vision framework landmarks
    /// Landmarks are returned in face bounding box coordinates (0-1 relative to bbox)
    /// We convert them to full image coordinates
    private func extractLandmarks(from visionLandmarks: VNFaceLandmarks2D, in boundingBox: CGRect) -> FaceLandmarks? {
        // Get key landmark regions
        guard let leftEyeRegion = visionLandmarks.leftEye,
              let rightEyeRegion = visionLandmarks.rightEye,
              let noseRegion = visionLandmarks.nose,
              let outerLipsRegion = visionLandmarks.outerLips else {
            return nil
        }

        // Compute center of each landmark region
        // Landmark points are in face bounding box coordinates (0-1)
        let leftEyeCenter = centerOfRegion(leftEyeRegion)
        let rightEyeCenter = centerOfRegion(rightEyeRegion)
        let noseCenter = centerOfRegion(noseRegion)

        // For mouth corners, get first and last point of outer lips
        let mouthPoints = outerLipsRegion.normalizedPoints
        guard mouthPoints.count >= 2 else { return nil }
        let leftMouth = mouthPoints[0]
        let rightMouth = mouthPoints[mouthPoints.count / 2] // Approximate right corner

        // Convert from face bbox coordinates to full image coordinates
        // Vision coordinates: origin at bottom-left, y increases upward
        func toImageCoords(_ point: CGPoint) -> CGPoint {
            return CGPoint(
                x: boundingBox.origin.x + point.x * boundingBox.width,
                y: boundingBox.origin.y + point.y * boundingBox.height
            )
        }

        return FaceLandmarks(
            leftEye: toImageCoords(leftEyeCenter),
            rightEye: toImageCoords(rightEyeCenter),
            nose: toImageCoords(noseCenter),
            leftMouth: toImageCoords(leftMouth),
            rightMouth: toImageCoords(rightMouth)
        )
    }

    /// Compute center point of a landmark region
    private func centerOfRegion(_ region: VNFaceLandmarkRegion2D) -> CGPoint {
        let points = region.normalizedPoints
        guard !points.isEmpty else { return .zero }

        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        for point in points {
            sumX += point.x
            sumY += point.y
        }
        return CGPoint(
            x: sumX / CGFloat(points.count),
            y: sumY / CGFloat(points.count)
        )
    }
}

// MARK: - Errors

enum FaceDetectionError: Error, LocalizedError {
    case invalidImage
    case detectionFailed(Error)

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Could not process image for face detection"
        case .detectionFailed(let error):
            return "Face detection failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    /// Convert UIImage orientation to CGImagePropertyOrientation for Vision framework
    var cgImageOrientation: CGImagePropertyOrientation {
        switch imageOrientation {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }
}
