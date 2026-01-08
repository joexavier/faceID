import Photos
import UIKit

/// High-level manager for face recognition functionality.
///
/// This class coordinates all the components:
/// - Face detection (Vision framework)
/// - Face alignment (cropping/resizing to 112x112)
/// - Embedding extraction (MobileFaceNet CoreML)
/// - Centroid matching (classification)
/// - Photo library scanning
///
/// ## Typical Workflow
///
/// ### 1. Enrollment
/// ```swift
/// let manager = FaceRecognitionManager()
///
/// // Select photos with target person
/// let assets: [PHAsset] = // from photo picker
///
/// // Enroll - extracts faces, computes centroid
/// let person = try await manager.enrollPerson(
///     name: "John",
///     from: assets,
///     threshold: 1.0
/// )
/// ```
///
/// ### 2. Scanning
/// ```swift
/// // Scan library for matches
/// for await result in manager.scanLibrary(for: person) {
///     if result.hasMatches {
///         print("Found \(person.name) in photo!")
///     }
///     print("Progress: \(Int(result.progress * 100))%")
/// }
/// ```
///
/// ### 3. Single Photo Check
/// ```swift
/// let matches = try await manager.findMatches(in: photo, for: person)
/// ```
@MainActor
class FaceRecognitionManager: ObservableObject {

    // MARK: - Published State

    /// Currently enrolled people
    @Published private(set) var enrolledPeople: [EnrolledPerson] = []

    /// Current scanning progress (0-1)
    @Published private(set) var scanProgress: Float = 0

    /// Whether currently scanning
    @Published private(set) var isScanning: Bool = false

    /// Last error that occurred
    @Published private(set) var lastError: Error?

    // MARK: - Services

    private let faceDetector = FaceDetector()
    private let faceAligner = FaceAligner()
    private let embeddingService: FaceEmbeddingService?
    private let scanner = PhotoLibraryScanner()

    // MARK: - Initialization

    init() {
        self.embeddingService = try? FaceEmbeddingService()

        // Load saved enrolled people
        loadEnrolledPeople()
    }

    // MARK: - Enrollment

    /// Enroll a new person from selected photos
    ///
    /// - Parameters:
    ///   - name: Person's name
    ///   - assets: Photos containing the person's face
    ///   - threshold: Matching threshold (default 1.0)
    /// - Returns: The enrolled person
    func enrollPerson(name: String, from assets: [PHAsset], threshold: Float = 1.0) async throws -> EnrolledPerson {
        guard let embeddingService = embeddingService else {
            throw ManagerError.embeddingServiceUnavailable
        }

        guard !assets.isEmpty else {
            throw ManagerError.noPhotosProvided
        }

        var embeddings: [FaceEmbedding] = []

        // Process each photo
        for asset in assets {
            // Load image
            let image = try await scanner.loadImage(from: asset)

            // Detect faces
            let faces = try await faceDetector.detectFaces(in: image)

            guard let face = faces.first else {
                // Skip photos with no detected face
                continue
            }

            // Align face
            guard let alignedFace = faceAligner.alignFace(from: image, face: face) else {
                continue
            }

            // Extract embedding
            let vector = try await embeddingService.extractEmbedding(from: alignedFace)

            let embedding = FaceEmbedding(
                vector: vector,
                sourceImageId: asset.localIdentifier,
                boundingBox: face.boundingBox
            )
            embeddings.append(embedding)
        }

        guard !embeddings.isEmpty else {
            throw ManagerError.noFacesDetected
        }

        // Create enrolled person
        let person = EnrolledPerson(
            name: name,
            embeddings: embeddings,
            threshold: threshold
        )

        // Save
        enrolledPeople.append(person)
        saveEnrolledPeople()

        return person
    }

    /// Remove an enrolled person
    func removePerson(_ person: EnrolledPerson) {
        enrolledPeople.removeAll { $0.id == person.id }
        saveEnrolledPeople()
    }

    /// Update threshold for an enrolled person
    func updateThreshold(for person: EnrolledPerson, to newThreshold: Float) {
        if let index = enrolledPeople.firstIndex(where: { $0.id == person.id }) {
            var updated = enrolledPeople[index]
            updated.threshold = newThreshold
            enrolledPeople[index] = updated
            saveEnrolledPeople()
        }
    }

    // MARK: - Scanning

    /// Scan entire photo library for a person
    ///
    /// - Parameter person: The enrolled person to search for
    /// - Returns: AsyncStream of scan results
    func scanLibrary(for person: EnrolledPerson) -> AsyncStream<ScanResult> {
        // Create classifier from enrolled person
        let classifier = CentroidClassifier(
            centroid: person.centroid,
            threshold: person.threshold,
            sampleCount: person.sampleCount
        )

        isScanning = true
        scanProgress = 0

        return AsyncStream { continuation in
            Task {
                for await result in scanner.scanLibrary(classifier: classifier) {
                    await MainActor.run {
                        self.scanProgress = result.progress
                    }
                    continuation.yield(result)
                }

                await MainActor.run {
                    self.isScanning = false
                    self.scanProgress = 1.0
                }
                continuation.finish()
            }
        }
    }

    /// Find all matches in a single photo
    func findMatches(in asset: PHAsset, for person: EnrolledPerson) async throws -> [MatchResult] {
        let classifier = CentroidClassifier(
            centroid: person.centroid,
            threshold: person.threshold,
            sampleCount: person.sampleCount
        )

        let result = try await scanner.scanPhoto(asset: asset, classifier: classifier)

        return result.matches.map { match in
            MatchResult(
                asset: asset,
                boundingBox: match.face.boundingBox,
                distance: match.distance,
                person: person
            )
        }
    }

    /// Check if a face matches an enrolled person
    func checkMatch(embedding: [Float], against person: EnrolledPerson) -> PredictionResult {
        let classifier = CentroidClassifier(
            centroid: person.centroid,
            threshold: person.threshold,
            sampleCount: person.sampleCount
        )
        return classifier.predict(embedding)
    }

    /// Extract face embedding from an image
    func extractEmbedding(from image: UIImage) async throws -> FaceEmbedding? {
        guard let embeddingService = embeddingService else {
            throw ManagerError.embeddingServiceUnavailable
        }

        // Detect face
        let faces = try await faceDetector.detectFaces(in: image)
        guard let face = faces.first else {
            return nil
        }

        // Align
        guard let alignedFace = faceAligner.alignFace(from: image, face: face) else {
            return nil
        }

        // Extract embedding
        let vector = try await embeddingService.extractEmbedding(from: alignedFace)

        return FaceEmbedding(
            vector: vector,
            sourceImageId: UUID().uuidString,
            boundingBox: face.boundingBox
        )
    }

    // MARK: - Persistence

    private let enrolledPeopleKey = "enrolledPeople"

    private func saveEnrolledPeople() {
        do {
            let data = try JSONEncoder().encode(enrolledPeople)
            UserDefaults.standard.set(data, forKey: enrolledPeopleKey)
        } catch {
            lastError = error
        }
    }

    private func loadEnrolledPeople() {
        guard let data = UserDefaults.standard.data(forKey: enrolledPeopleKey) else {
            return
        }

        do {
            enrolledPeople = try JSONDecoder().decode([EnrolledPerson].self, from: data)
        } catch {
            lastError = error
        }
    }

    /// Export enrolled people to file (for backup)
    func exportEnrolledPeople(to url: URL) throws {
        let data = try JSONEncoder().encode(enrolledPeople)
        try data.write(to: url)
    }

    /// Import enrolled people from file
    func importEnrolledPeople(from url: URL) throws {
        let data = try Data(contentsOf: url)
        let imported = try JSONDecoder().decode([EnrolledPerson].self, from: data)
        enrolledPeople.append(contentsOf: imported)
        saveEnrolledPeople()
    }

    // MARK: - Permissions

    /// Request photo library permission
    func requestPhotoPermission() async -> Bool {
        return await scanner.requestPermission()
    }

    /// Current photo library permission status
    var photoPermissionStatus: PHAuthorizationStatus {
        return scanner.authorizationStatus
    }
}

// MARK: - Errors

enum ManagerError: Error, LocalizedError {
    case embeddingServiceUnavailable
    case noPhotosProvided
    case noFacesDetected
    case enrollmentFailed(String)

    var errorDescription: String? {
        switch self {
        case .embeddingServiceUnavailable:
            return "Face embedding model not available. Ensure MobileFaceNet.mlmodelc is in the app bundle."
        case .noPhotosProvided:
            return "No photos were provided for enrollment"
        case .noFacesDetected:
            return "No faces were detected in the provided photos"
        case .enrollmentFailed(let reason):
            return "Enrollment failed: \(reason)"
        }
    }
}

// MARK: - Convenience Extensions

extension FaceRecognitionManager {
    /// Quick check if anyone is enrolled
    var hasEnrolledPeople: Bool {
        return !enrolledPeople.isEmpty
    }

    /// Get enrolled person by name
    func getPerson(named name: String) -> EnrolledPerson? {
        return enrolledPeople.first { $0.name == name }
    }

    /// Get statistics about enrolled person
    func getStats(for person: EnrolledPerson) -> String {
        return """
        Name: \(person.name)
        Samples: \(person.sampleCount)
        Threshold: \(String(format: "%.2f", person.threshold))
        Enrolled: \(person.enrolledAt.formatted())
        """
    }
}
