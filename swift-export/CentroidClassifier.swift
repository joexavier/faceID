import Foundation

/// The Centroid Classifier is the core algorithm for face matching.
///
/// ## How It Works
///
/// ### Training (Enrollment)
/// 1. User provides N face images of the target person (recommended: 5-10)
/// 2. Extract embedding from each face (512-dimensional vector)
/// 3. Compute centroid: average of all embeddings
/// 4. L2-normalize the centroid to unit length
///
/// ### Matching
/// 1. Extract embedding from candidate face
/// 2. Compute Euclidean distance to centroid
/// 3. If distance < threshold, it's a match
///
/// ## Distance Interpretation
/// - **Same person**: typically distance < 0.8
/// - **Different person**: typically distance > 1.0
/// - The threshold can be tuned based on your false positive/negative tolerance
///
/// ## Example
/// ```swift
/// let classifier = CentroidClassifier()
///
/// // Train with embeddings
/// let trainingEmbeddings: [[Float]] = // extracted from user's photos
/// classifier.train(with: trainingEmbeddings)
///
/// // Match a new face
/// let candidateEmbedding: [Float] = // extracted from photo to test
/// let result = classifier.predict(candidateEmbedding)
/// if result.isMatch {
///     print("Match found! Distance: \(result.distance)")
/// }
/// ```
class CentroidClassifier: Codable {

    // MARK: - Properties

    /// The centroid embedding (average of training samples, L2-normalized)
    /// This is the "master template" representing the enrolled person
    private(set) var centroid: [Float]?

    /// Distance threshold for matching
    /// Faces with distance < threshold are considered matches
    /// Default: 1.0 (balanced false positives/negatives)
    var threshold: Float = 1.0

    /// Number of samples used to compute the centroid
    private(set) var sampleCount: Int = 0

    /// Whether the classifier has been trained
    var isTrained: Bool {
        return centroid != nil
    }

    // MARK: - Initialization

    init() {}

    /// Initialize with a pre-computed centroid (for loading saved models)
    init(centroid: [Float], threshold: Float, sampleCount: Int) {
        self.centroid = centroid
        self.threshold = threshold
        self.sampleCount = sampleCount
    }

    // MARK: - Training

    /// Train the classifier with face embeddings
    ///
    /// This computes the centroid (average) of all embeddings and L2-normalizes it.
    /// More diverse training samples (different angles, lighting) produce better results.
    ///
    /// - Parameter embeddings: Array of 512-dimensional face embeddings
    /// - Returns: Training statistics including threshold recommendation
    @discardableResult
    func train(with embeddings: [[Float]]) -> TrainingResult {
        guard !embeddings.isEmpty else {
            return TrainingResult(success: false, error: "No embeddings provided")
        }

        let dim = embeddings[0].count
        guard dim == 512 else {
            return TrainingResult(success: false, error: "Embeddings must be 512-dimensional")
        }

        // Step 1: Compute mean (centroid)
        var mean = [Float](repeating: 0, count: dim)
        for embedding in embeddings {
            for i in 0..<dim {
                mean[i] += embedding[i]
            }
        }
        let count = Float(embeddings.count)
        for i in 0..<dim {
            mean[i] /= count
        }

        // Step 2: L2 normalize the centroid
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

        self.centroid = mean
        self.sampleCount = embeddings.count

        // Step 3: Calculate statistics for threshold recommendation
        var distances = [Float]()
        for embedding in embeddings {
            let dist = Self.euclideanDistance(embedding, mean)
            distances.append(dist)
        }

        let maxDistance = distances.max() ?? 0
        let avgDistance = distances.reduce(0, +) / Float(distances.count)

        // Recommended threshold: max training distance * 1.5 (with some margin)
        let recommendedThreshold = maxDistance * 1.5
        self.threshold = recommendedThreshold

        return TrainingResult(
            success: true,
            sampleCount: embeddings.count,
            averageDistance: avgDistance,
            maxDistance: maxDistance,
            recommendedThreshold: recommendedThreshold
        )
    }

    /// Train with negative examples to optimize threshold
    ///
    /// Uses both positive (target person) and negative (other people) examples
    /// to find the optimal threshold that maximizes F1 score.
    ///
    /// - Parameters:
    ///   - positiveEmbeddings: Embeddings of the target person
    ///   - negativeEmbeddings: Embeddings of other people (not target)
    /// - Returns: Training result with optimized threshold
    @discardableResult
    func train(positives positiveEmbeddings: [[Float]],
               negatives negativeEmbeddings: [[Float]]) -> TrainingResult {

        // First, train normally with positives
        let result = train(with: positiveEmbeddings)
        guard result.success, let centroid = self.centroid else {
            return result
        }

        // If no negatives, return basic result
        guard !negativeEmbeddings.isEmpty else {
            return result
        }

        // Calculate distances for all samples
        let posDistances = positiveEmbeddings.map { Self.euclideanDistance($0, centroid) }
        let negDistances = negativeEmbeddings.map { Self.euclideanDistance($0, centroid) }

        // Find optimal threshold using F1 score
        let maxDist = max(posDistances.max() ?? 0, negDistances.max() ?? 0)
        var bestF1: Float = 0
        var bestThreshold = maxDist * 1.2

        // Search thresholds from 0.3 to maxDist * 1.5
        stride(from: Float(0.3), through: maxDist * 1.5, by: 0.05).forEach { thresh in
            // True positives: positives below threshold
            let tp = posDistances.filter { $0 < thresh }.count
            // False positives: negatives below threshold
            let fp = negDistances.filter { $0 < thresh }.count
            // False negatives: positives above threshold
            let fn = posDistances.filter { $0 >= thresh }.count

            let precision = tp + fp > 0 ? Float(tp) / Float(tp + fp) : 0
            let recall = tp + fn > 0 ? Float(tp) / Float(tp + fn) : 0
            let f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0

            if f1 > bestF1 {
                bestF1 = f1
                bestThreshold = thresh
            }
        }

        self.threshold = bestThreshold

        return TrainingResult(
            success: true,
            sampleCount: positiveEmbeddings.count,
            averageDistance: posDistances.reduce(0, +) / Float(posDistances.count),
            maxDistance: posDistances.max() ?? 0,
            recommendedThreshold: bestThreshold,
            f1Score: bestF1
        )
    }

    // MARK: - Prediction

    /// Predict whether an embedding matches the enrolled person
    ///
    /// - Parameter embedding: 512-dimensional face embedding to test
    /// - Returns: Prediction result with match status and distance
    func predict(_ embedding: [Float]) -> PredictionResult {
        guard let centroid = centroid else {
            return PredictionResult(isMatch: false, distance: Float.infinity, error: "Classifier not trained")
        }

        let distance = Self.euclideanDistance(embedding, centroid)
        let isMatch = distance < threshold

        return PredictionResult(
            isMatch: isMatch,
            distance: distance,
            threshold: threshold
        )
    }

    /// Predict with detailed confidence levels
    func predictDetailed(_ embedding: [Float]) -> DetailedPrediction {
        guard let centroid = centroid else {
            return DetailedPrediction(confidence: .none, distance: Float.infinity)
        }

        let distance = Self.euclideanDistance(embedding, centroid)

        let confidence: MatchConfidence
        if distance < 0.7 {
            confidence = .high
        } else if distance < 1.0 {
            confidence = .medium
        } else if distance < 1.3 {
            confidence = .low
        } else {
            confidence = .none
        }

        return DetailedPrediction(confidence: confidence, distance: distance)
    }

    // MARK: - Utility Methods

    /// Euclidean distance between two vectors
    /// Formula: sqrt(sum((a[i] - b[i])^2))
    static func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count, "Vectors must have same dimension")
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    /// Cosine similarity between two vectors
    /// For L2-normalized vectors, this equals the dot product
    /// Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count, "Vectors must have same dimension")
        var dot: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
        }
        return dot
    }

    /// Update threshold manually
    func setThreshold(_ newThreshold: Float) {
        self.threshold = newThreshold
    }

    // MARK: - Persistence

    /// Save classifier to file
    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    /// Load classifier from file
    static func load(from url: URL) throws -> CentroidClassifier {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(CentroidClassifier.self, from: data)
    }
}

// MARK: - Result Types

/// Result of training the classifier
struct TrainingResult {
    let success: Bool
    var error: String?
    var sampleCount: Int = 0
    var averageDistance: Float = 0
    var maxDistance: Float = 0
    var recommendedThreshold: Float = 0
    var f1Score: Float?

    init(success: Bool, error: String? = nil) {
        self.success = success
        self.error = error
    }

    init(success: Bool, sampleCount: Int, averageDistance: Float,
         maxDistance: Float, recommendedThreshold: Float, f1Score: Float? = nil) {
        self.success = success
        self.sampleCount = sampleCount
        self.averageDistance = averageDistance
        self.maxDistance = maxDistance
        self.recommendedThreshold = recommendedThreshold
        self.f1Score = f1Score
    }
}

/// Result of predicting a single embedding
struct PredictionResult {
    let isMatch: Bool
    let distance: Float
    var threshold: Float = 0
    var error: String?
}

/// Detailed prediction with confidence level
struct DetailedPrediction {
    let confidence: MatchConfidence
    let distance: Float

    var isMatch: Bool {
        return confidence != .none
    }
}

/// Match confidence levels based on distance
enum MatchConfidence: String, Codable {
    case high   // distance < 0.7: very confident match
    case medium // distance 0.7-1.0: likely match
    case low    // distance 1.0-1.3: possible match
    case none   // distance >= 1.3: not a match
}
