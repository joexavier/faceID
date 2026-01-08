import CoreML
import Vision
import UIKit

/// Service for extracting face embeddings using the MobileFaceNet CoreML model.
///
/// The embedding is a 512-dimensional vector that represents the "identity" of a face.
/// Two faces of the same person will have similar embeddings (low Euclidean distance),
/// while different people will have dissimilar embeddings (high distance).
///
/// ## Model Requirements
/// - **Input**: 112x112 RGB image
/// - **Normalization**: Pixel values mapped from [0, 255] to [-1, 1]
///   Formula: normalized = (pixel - 127.5) / 127.5
/// - **Output**: 512-dimensional float vector (L2-normalized, magnitude = 1.0)
///
/// ## Usage
/// ```swift
/// let service = try FaceEmbeddingService()
/// let embedding = try await service.extractEmbedding(from: alignedFaceImage)
/// ```
class FaceEmbeddingService {

    // MARK: - Properties

    /// The CoreML model for face embedding extraction
    /// This should be the MobileFaceNet.mlmodelc compiled model
    private let model: MLModel

    /// Compute units to use (prefer Neural Engine for speed)
    private static let computeUnits: MLComputeUnits = .all

    // MARK: - Initialization

    /// Initialize with the bundled MobileFaceNet model
    /// - Throws: If model cannot be loaded
    init() throws {
        // Load the model with configuration
        let config = MLModelConfiguration()
        config.computeUnits = Self.computeUnits

        // Option 1: Load from auto-generated class (if MobileFaceNet.mlmodelc is in bundle)
        // self.model = try MobileFaceNet(configuration: config).model

        // Option 2: Load from URL (more flexible)
        guard let modelURL = Bundle.main.url(forResource: "MobileFaceNet", withExtension: "mlmodelc") else {
            throw EmbeddingError.modelNotFound
        }
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }

    /// Initialize with a custom model URL
    /// - Parameter modelURL: URL to the .mlmodelc compiled model
    init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = Self.computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }

    // MARK: - Public API

    /// Extract embedding from an aligned face image
    /// - Parameter image: 112x112 aligned face image
    /// - Returns: 512-dimensional embedding
    func extractEmbedding(from image: UIImage) async throws -> [Float] {
        // Ensure image is correct size
        guard image.size.width == 112 && image.size.height == 112 else {
            throw EmbeddingError.invalidImageSize
        }

        // Convert to pixel buffer
        guard let pixelBuffer = createPixelBuffer(from: image) else {
            throw EmbeddingError.pixelBufferCreationFailed
        }

        return try await extractEmbedding(from: pixelBuffer)
    }

    /// Extract embedding from a pixel buffer
    /// - Parameter pixelBuffer: 112x112 pixel buffer
    /// - Returns: 512-dimensional embedding
    func extractEmbedding(from pixelBuffer: CVPixelBuffer) async throws -> [Float] {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let embedding = try self.runInference(on: pixelBuffer)
                    continuation.resume(returning: embedding)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Extract embedding using Vision framework (handles preprocessing automatically)
    /// - Parameter image: Face image (will be resized to 112x112)
    /// - Returns: 512-dimensional embedding
    func extractEmbeddingWithVision(from image: UIImage) async throws -> [Float] {
        guard let cgImage = image.cgImage else {
            throw EmbeddingError.invalidImage
        }

        // Create Vision CoreML request
        guard let visionModel = try? VNCoreMLModel(for: model) else {
            throw EmbeddingError.visionModelCreationFailed
        }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: visionModel) { request, error in
                if let error = error {
                    continuation.resume(throwing: EmbeddingError.inferenceFailed(error))
                    return
                }

                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let firstResult = results.first,
                      let multiArray = firstResult.featureValue.multiArrayValue else {
                    continuation.resume(throwing: EmbeddingError.invalidOutput)
                    return
                }

                // Convert MLMultiArray to [Float]
                let embedding = self.multiArrayToFloatArray(multiArray)
                continuation.resume(returning: embedding)
            }

            // Configure preprocessing (resize to 112x112, normalize to [-1, 1])
            request.imageCropAndScaleOption = .scaleFill

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])
                } catch {
                    continuation.resume(throwing: EmbeddingError.inferenceFailed(error))
                }
            }
        }
    }

    // MARK: - Private Methods

    /// Run inference directly on pixel buffer
    private func runInference(on pixelBuffer: CVPixelBuffer) throws -> [Float] {
        // Create input feature provider
        // The input name depends on your model - check the .mlmodel file
        // Common names: "input", "image", "data"
        let inputName = "input"  // Adjust based on your model

        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])

        // Run prediction
        let output = try model.prediction(from: input)

        // Get output embedding
        // The output name depends on your model - check the .mlmodel file
        // Common names: "output", "embedding", "fc1"
        guard let outputFeature = output.featureValue(for: output.featureNames.first ?? "output"),
              let multiArray = outputFeature.multiArrayValue else {
            throw EmbeddingError.invalidOutput
        }

        // Convert to float array
        var embedding = multiArrayToFloatArray(multiArray)

        // L2 normalize the embedding (if not already normalized by the model)
        embedding = l2Normalize(embedding)

        return embedding
    }

    /// Convert MLMultiArray to [Float]
    private func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var result = [Float](repeating: 0, count: count)

        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = ptr[i]
        }

        return result
    }

    /// L2 normalize a vector to unit length
    /// This ensures cosine similarity = dot product
    private func l2Normalize(_ vector: [Float]) -> [Float] {
        var norm: Float = 0
        for v in vector {
            norm += v * v
        }
        norm = sqrt(norm)

        guard norm > 0 else { return vector }

        return vector.map { $0 / norm }
    }

    /// Create pixel buffer from UIImage with proper format for the model
    private func createPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }

        let width = 112
        let height = 112

        // Create pixel buffer attributes
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true
        ]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,  // Common format for CoreML
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        // Lock buffer and draw image
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            return nil
        }

        // Draw image (this also handles resizing if needed)
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}

// MARK: - Errors

enum EmbeddingError: Error, LocalizedError {
    case modelNotFound
    case invalidImage
    case invalidImageSize
    case pixelBufferCreationFailed
    case visionModelCreationFailed
    case inferenceFailed(Error)
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "MobileFaceNet.mlmodelc not found in bundle"
        case .invalidImage:
            return "Could not process image"
        case .invalidImageSize:
            return "Image must be 112x112 pixels"
        case .pixelBufferCreationFailed:
            return "Failed to create pixel buffer"
        case .visionModelCreationFailed:
            return "Failed to create Vision CoreML model"
        case .inferenceFailed(let error):
            return "Model inference failed: \(error.localizedDescription)"
        case .invalidOutput:
            return "Model returned invalid output"
        }
    }
}
