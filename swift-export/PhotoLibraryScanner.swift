import Photos
import UIKit

/// Service for scanning the Photo Library and finding faces.
///
/// ## Permissions
/// Add to Info.plist:
/// ```xml
/// <key>NSPhotoLibraryUsageDescription</key>
/// <string>We need access to scan your photos for faces.</string>
/// ```
///
/// ## Usage
/// ```swift
/// let scanner = PhotoLibraryScanner()
///
/// // Request permission
/// let granted = await scanner.requestPermission()
/// guard granted else { return }
///
/// // Scan all photos
/// for await result in scanner.scanLibrary() {
///     print("Found face in \(result.asset.localIdentifier)")
/// }
/// ```
class PhotoLibraryScanner {

    // MARK: - Dependencies

    private let faceDetector = FaceDetector()
    private let faceAligner = FaceAligner()
    private let embeddingService: FaceEmbeddingService?

    // MARK: - Configuration

    /// Maximum image dimension for face detection (resize larger images)
    private let maxImageDimension: CGFloat = 1024

    /// Batch size for processing photos
    private let batchSize: Int = 50

    // MARK: - Initialization

    init() {
        // Try to load embedding service (may fail if model not available)
        self.embeddingService = try? FaceEmbeddingService()
    }

    // MARK: - Permission Handling

    /// Current photo library authorization status
    var authorizationStatus: PHAuthorizationStatus {
        return PHPhotoLibrary.authorizationStatus(for: .readWrite)
    }

    /// Request photo library access permission
    /// - Returns: true if access granted
    func requestPermission() async -> Bool {
        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        return status == .authorized || status == .limited
    }

    // MARK: - Fetching Photos

    /// Fetch all photos from library
    func fetchAllPhotos() -> PHFetchResult<PHAsset> {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.predicate = NSPredicate(format: "mediaType = %d", PHAssetMediaType.image.rawValue)
        return PHAsset.fetchAssets(with: options)
    }

    /// Fetch photos from a specific album
    func fetchPhotos(from album: PHAssetCollection) -> PHFetchResult<PHAsset> {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.predicate = NSPredicate(format: "mediaType = %d", PHAssetMediaType.image.rawValue)
        return PHAsset.fetchAssets(in: album, options: options)
    }

    /// Fetch all user albums
    func fetchAlbums() -> [PHAssetCollection] {
        var albums: [PHAssetCollection] = []

        // Smart albums (Favorites, Recently Added, etc.)
        let smartAlbums = PHAssetCollection.fetchAssetCollections(
            with: .smartAlbum,
            subtype: .any,
            options: nil
        )
        smartAlbums.enumerateObjects { collection, _, _ in
            if PHAsset.fetchAssets(in: collection, options: nil).count > 0 {
                albums.append(collection)
            }
        }

        // User-created albums
        let userAlbums = PHAssetCollection.fetchAssetCollections(
            with: .album,
            subtype: .any,
            options: nil
        )
        userAlbums.enumerateObjects { collection, _, _ in
            albums.append(collection)
        }

        return albums
    }

    // MARK: - Image Loading

    /// Load UIImage from PHAsset
    func loadImage(from asset: PHAsset, targetSize: CGSize? = nil) async throws -> UIImage {
        return try await withCheckedThrowingContinuation { continuation in
            let options = PHImageRequestOptions()
            options.deliveryMode = .highQualityFormat
            options.isNetworkAccessAllowed = true
            options.isSynchronous = false

            let size = targetSize ?? CGSize(width: asset.pixelWidth, height: asset.pixelHeight)

            PHImageManager.default().requestImage(
                for: asset,
                targetSize: size,
                contentMode: .aspectFit,
                options: options
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                } else if let image = image {
                    continuation.resume(returning: image)
                } else {
                    continuation.resume(throwing: ScannerError.imageLoadFailed)
                }
            }
        }
    }

    /// Load image at appropriate size for face detection
    private func loadImageForDetection(from asset: PHAsset) async throws -> UIImage {
        // Calculate target size (scale down large images)
        let scale = min(1.0, maxImageDimension / max(CGFloat(asset.pixelWidth), CGFloat(asset.pixelHeight)))
        let targetSize = CGSize(
            width: CGFloat(asset.pixelWidth) * scale,
            height: CGFloat(asset.pixelHeight) * scale
        )
        return try await loadImage(from: asset, targetSize: targetSize)
    }

    // MARK: - Scanning

    /// Scan all photos in library and yield results as async stream
    ///
    /// - Parameter classifier: Optional classifier to filter matches
    /// - Returns: AsyncStream of scan results
    func scanLibrary(classifier: CentroidClassifier? = nil) -> AsyncStream<ScanResult> {
        return AsyncStream { continuation in
            Task {
                let assets = fetchAllPhotos()
                let total = assets.count

                for i in 0..<total {
                    let asset = assets.object(at: i)

                    do {
                        let result = try await scanPhoto(asset: asset, classifier: classifier)
                        continuation.yield(ScanResult(
                            asset: asset,
                            index: i,
                            total: total,
                            faces: result.faces,
                            matches: result.matches,
                            error: nil
                        ))
                    } catch {
                        continuation.yield(ScanResult(
                            asset: asset,
                            index: i,
                            total: total,
                            faces: [],
                            matches: [],
                            error: error
                        ))
                    }

                    // Small delay to prevent overwhelming the system
                    if i % batchSize == 0 {
                        try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
                    }
                }

                continuation.finish()
            }
        }
    }

    /// Scan a single photo for faces
    func scanPhoto(asset: PHAsset, classifier: CentroidClassifier? = nil) async throws -> PhotoScanResult {
        // Load image
        let image = try await loadImageForDetection(from: asset)

        // Detect faces
        let detectedFaces = try await faceDetector.detectFaces(in: image)

        var faces: [ScannedFace] = []
        var matches: [FaceMatch] = []

        for face in detectedFaces {
            // Align face for embedding extraction
            guard let alignedFace = faceAligner.alignFace(from: image, face: face) else {
                continue
            }

            // Extract embedding if service available
            var embedding: [Float]? = nil
            if let service = embeddingService {
                embedding = try? await service.extractEmbedding(from: alignedFace)
            }

            let scannedFace = ScannedFace(
                boundingBox: face.boundingBox,
                confidence: face.confidence,
                embedding: embedding
            )
            faces.append(scannedFace)

            // Check if matches classifier
            if let classifier = classifier,
               let emb = embedding {
                let prediction = classifier.predict(emb)
                if prediction.isMatch {
                    matches.append(FaceMatch(
                        face: scannedFace,
                        distance: prediction.distance
                    ))
                }
            }
        }

        return PhotoScanResult(faces: faces, matches: matches)
    }

    /// Scan multiple specific assets
    func scanPhotos(_ assets: [PHAsset], classifier: CentroidClassifier? = nil) async -> [ScanResult] {
        var results: [ScanResult] = []
        let total = assets.count

        for (i, asset) in assets.enumerated() {
            do {
                let result = try await scanPhoto(asset: asset, classifier: classifier)
                results.append(ScanResult(
                    asset: asset,
                    index: i,
                    total: total,
                    faces: result.faces,
                    matches: result.matches,
                    error: nil
                ))
            } catch {
                results.append(ScanResult(
                    asset: asset,
                    index: i,
                    total: total,
                    faces: [],
                    matches: [],
                    error: error
                ))
            }
        }

        return results
    }
}

// MARK: - Result Types

/// Result from scanning a single photo
struct PhotoScanResult {
    let faces: [ScannedFace]
    let matches: [FaceMatch]
}

/// A face found during scanning
struct ScannedFace {
    let boundingBox: CGRect
    let confidence: Float
    let embedding: [Float]?
}

/// A face that matched the classifier
struct FaceMatch {
    let face: ScannedFace
    let distance: Float
}

/// Result yielded during library scan
struct ScanResult {
    let asset: PHAsset
    let index: Int
    let total: Int
    let faces: [ScannedFace]
    let matches: [FaceMatch]
    let error: Error?

    var progress: Float {
        return Float(index + 1) / Float(total)
    }

    var hasMatches: Bool {
        return !matches.isEmpty
    }
}

// MARK: - Errors

enum ScannerError: Error, LocalizedError {
    case permissionDenied
    case imageLoadFailed
    case scanFailed(Error)

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Photo library access denied"
        case .imageLoadFailed:
            return "Failed to load image from photo library"
        case .scanFailed(let error):
            return "Scan failed: \(error.localizedDescription)"
        }
    }
}
