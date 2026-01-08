# iOS Face Recognition - Centroid Algorithm

This folder contains Swift code to implement face recognition using the Centroid matching algorithm. Use these files as a reference to build an iOS app that can identify a specific person across the Photo Library.

## Overview

The system works in three phases:

1. **Enrollment**: User selects 5-10 photos of the target person. The app extracts face embeddings and computes a "centroid" (average) embedding.

2. **Matching**: For any new face, compute its embedding and measure the Euclidean distance to the centroid. Lower distance = better match.

3. **Scanning**: Iterate through the Photo Library, detect faces, extract embeddings, and find matches.

## Required Model

You need the **MobileFaceNet CoreML model** (`MobileFaceNet.mlpackage`):
- Input: 112x112 RGB image, normalized to [-1, 1]
- Output: 512-dimensional L2-normalized embedding
- Source: InsightFace buffalo_s pack (w600k_mbf.onnx converted to CoreML)

Place the model in your Xcode project and it will auto-generate `MobileFaceNet.swift`.

## Files

| File | Purpose |
|------|---------|
| `Models.swift` | Data structures (FaceEmbedding, EnrolledPerson, MatchResult) |
| `FaceDetector.swift` | Vision framework face detection with bounding boxes |
| `FaceAligner.swift` | Align/crop faces to 112x112 for the model |
| `FaceEmbeddingService.swift` | CoreML inference to extract 512-D embeddings |
| `CentroidClassifier.swift` | Centroid matching algorithm (the core logic) |
| `PhotoLibraryScanner.swift` | Scan Photo Library for faces |
| `FaceRecognitionManager.swift` | High-level API coordinating all components |

## Key Algorithm: Centroid Matching

```swift
// Enrollment: Compute centroid from N face embeddings
let centroid = embeddings.reduce(into: [Float](repeating: 0, count: 512)) { result, emb in
    for i in 0..<512 { result[i] += emb[i] }
}.map { $0 / Float(embeddings.count) }

// Matching: Euclidean distance to centroid
let distance = sqrt(zip(embedding, centroid).map { pow($0 - $1, 2) }.reduce(0, +))
let isMatch = distance < threshold  // threshold ~0.8-1.0
```

## Usage Example

```swift
let manager = FaceRecognitionManager()

// 1. Enroll target person with 5 photos
let assets: [PHAsset] = // user-selected photos
await manager.enrollPerson(name: "John", from: assets)

// 2. Scan library for matches
let matches = await manager.scanLibrary(threshold: 1.0)
for match in matches {
    print("\(match.asset.localIdentifier): distance=\(match.distance)")
}
```

## Thresholds

Based on testing with the InsightFace model:
- `distance < 0.8`: High confidence match
- `distance < 1.0`: Probable match
- `distance >= 1.0`: Unlikely match

Adjust based on your false positive/negative tolerance.

## Privacy

- All processing is on-device (no server)
- Request Photo Library permission: `NSPhotoLibraryUsageDescription`
- Store enrolled embeddings securely (Keychain recommended)
