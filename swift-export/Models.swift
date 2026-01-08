import Foundation
import Photos

// MARK: - Face Embedding
/// Represents a face embedding extracted from an image.
/// The embedding is a 512-dimensional vector that captures facial features.
/// Two embeddings can be compared using Euclidean distance - lower distance means more similar faces.
struct FaceEmbedding: Codable {
    /// The 512-dimensional embedding vector (L2-normalized, so magnitude = 1.0)
    let vector: [Float]

    /// Unique identifier for this embedding
    let id: UUID

    /// Source image identifier (PHAsset localIdentifier or file path)
    let sourceImageId: String

    /// Bounding box of the face in the original image (normalized 0-1 coordinates)
    let boundingBox: CGRect

    /// Timestamp when embedding was extracted
    let createdAt: Date

    init(vector: [Float], sourceImageId: String, boundingBox: CGRect) {
        // Validate embedding dimension
        assert(vector.count == 512, "Embedding must be 512-dimensional")

        self.vector = vector
        self.id = UUID()
        self.sourceImageId = sourceImageId
        self.boundingBox = boundingBox
        self.createdAt = Date()
    }

    // MARK: - Distance Calculations

    /// Compute Euclidean distance to another embedding.
    /// Lower distance = more similar faces.
    /// Same person typically has distance < 1.0
    /// Different people typically have distance > 1.0
    func euclideanDistance(to other: FaceEmbedding) -> Float {
        return Self.euclideanDistance(self.vector, other.vector)
    }

    /// Static method to compute Euclidean distance between two vectors
    static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count, "Vectors must have same dimension")
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    /// Compute cosine similarity to another embedding.
    /// For L2-normalized vectors, this equals the dot product.
    /// Range: -1 to 1, higher = more similar
    func cosineSimilarity(to other: FaceEmbedding) -> Float {
        return Self.cosineSimilarity(self.vector, other.vector)
    }

    /// Static method to compute cosine similarity (dot product for L2-normalized vectors)
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count, "Vectors must have same dimension")
        var dot: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
        }
        return dot
    }
}

// MARK: - Enrolled Person
/// Represents a person enrolled in the face recognition system.
/// Contains the centroid embedding computed from multiple face samples.
struct EnrolledPerson: Codable {
    /// Person's name
    let name: String

    /// Unique identifier
    let id: UUID

    /// The centroid (average) embedding computed from enrollment samples.
    /// This is the "master template" used for matching.
    /// Computed as: centroid = normalize(mean(all_embeddings))
    let centroid: [Float]

    /// Number of samples used to compute the centroid
    let sampleCount: Int

    /// Matching threshold - faces with distance < threshold are considered matches
    /// Recommended: 0.8-1.0 for InsightFace model
    var threshold: Float

    /// IDs of source images used for enrollment (for reference)
    let sourceImageIds: [String]

    /// When this person was enrolled
    let enrolledAt: Date

    init(name: String, embeddings: [FaceEmbedding], threshold: Float = 1.0) {
        assert(!embeddings.isEmpty, "Need at least one embedding to enroll")

        self.name = name
        self.id = UUID()
        self.sampleCount = embeddings.count
        self.threshold = threshold
        self.sourceImageIds = embeddings.map { $0.sourceImageId }
        self.enrolledAt = Date()

        // Compute centroid: average of all embeddings, then L2-normalize
        self.centroid = Self.computeCentroid(from: embeddings.map { $0.vector })
    }

    /// Compute centroid from multiple embedding vectors.
    /// Steps:
    /// 1. Sum all vectors element-wise
    /// 2. Divide by count to get mean
    /// 3. L2-normalize to unit length
    static func computeCentroid(from vectors: [[Float]]) -> [Float] {
        guard !vectors.isEmpty else { return [] }
        let dim = vectors[0].count

        // Step 1 & 2: Compute mean
        var mean = [Float](repeating: 0, count: dim)
        for vector in vectors {
            for i in 0..<dim {
                mean[i] += vector[i]
            }
        }
        let count = Float(vectors.count)
        for i in 0..<dim {
            mean[i] /= count
        }

        // Step 3: L2-normalize
        var norm: Float = 0
        for i in 0..<dim {
            norm += mean[i] * mean[i]
        }
        norm = sqrt(norm)

        if norm > 0 {
            for i in 0..<dim {
                mean[i] /= norm
            }
        }

        return mean
    }

    /// Check if a face embedding matches this person
    func matches(_ embedding: FaceEmbedding) -> Bool {
        let distance = FaceEmbedding.euclideanDistance(embedding.vector, centroid)
        return distance < threshold
    }

    /// Get match score (distance) for an embedding
    func matchScore(_ embedding: FaceEmbedding) -> Float {
        return FaceEmbedding.euclideanDistance(embedding.vector, centroid)
    }
}

// MARK: - Match Result
/// Result of matching a face against an enrolled person
struct MatchResult {
    /// The photo asset containing the matched face
    let asset: PHAsset

    /// Bounding box of the matched face in the image
    let boundingBox: CGRect

    /// Distance to the enrolled person's centroid (lower = better match)
    let distance: Float

    /// The enrolled person this face matched
    let person: EnrolledPerson

    /// Confidence level based on distance
    var confidence: MatchConfidence {
        if distance < 0.7 {
            return .high
        } else if distance < 1.0 {
            return .medium
        } else {
            return .low
        }
    }
}

/// Match confidence levels
enum MatchConfidence {
    case high    // distance < 0.7 - very likely same person
    case medium  // distance 0.7-1.0 - probable match
    case low     // distance >= 1.0 - unlikely match
}

// MARK: - Detected Face
/// Represents a face detected in an image (before embedding extraction)
struct DetectedFace {
    /// Bounding box in normalized coordinates (0-1)
    let boundingBox: CGRect

    /// Face landmarks if available (eyes, nose, mouth positions)
    let landmarks: FaceLandmarks?

    /// Confidence score from face detector (0-1)
    let confidence: Float
}

/// Facial landmark positions for alignment
struct FaceLandmarks {
    /// Left eye center (normalized coordinates)
    let leftEye: CGPoint

    /// Right eye center (normalized coordinates)
    let rightEye: CGPoint

    /// Nose tip (normalized coordinates)
    let nose: CGPoint

    /// Left mouth corner (normalized coordinates)
    let leftMouth: CGPoint

    /// Right mouth corner (normalized coordinates)
    let rightMouth: CGPoint
}
