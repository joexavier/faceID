from flask import Blueprint, jsonify, request, current_app, send_file
import os
import hashlib
from datetime import datetime

from app import db
from app.models.database import Person, Photo, DetectedFace, FaceExample, Classifier, ScanResult
from app.services.face_detection import FaceDetectionService
from app.services.embedding_service import EmbeddingService
from app.services.diversity_analyzer import DiversityAnalyzer
from app.services.photo_scanner import PhotoScanner
from app.models.classifier import create_classifier, train_and_save_classifier, load_classifier

api_bp = Blueprint('api', __name__, url_prefix='/api')


# ============== Person APIs ==============

@api_bp.route('/persons', methods=['GET'])
def list_persons():
    """List all persons"""
    persons = Person.query.all()
    return jsonify([p.to_dict() for p in persons])


@api_bp.route('/persons', methods=['POST'])
def create_person():
    """Create a new person"""
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    person = Person(name=name, is_active=True)

    # Deactivate other persons
    Person.query.update({'is_active': False})

    db.session.add(person)
    db.session.commit()

    return jsonify(person.to_dict()), 201


@api_bp.route('/persons/<int:person_id>', methods=['GET'])
def get_person(person_id):
    """Get a person by ID"""
    person = Person.query.get_or_404(person_id)
    return jsonify(person.to_dict())


@api_bp.route('/persons/<int:person_id>/activate', methods=['POST'])
def activate_person(person_id):
    """Set a person as active"""
    Person.query.update({'is_active': False})
    person = Person.query.get_or_404(person_id)
    person.is_active = True
    db.session.commit()
    return jsonify(person.to_dict())


@api_bp.route('/persons/active', methods=['GET'])
def get_active_person():
    """Get the currently active person"""
    person = Person.query.filter_by(is_active=True).first()
    if not person:
        return jsonify({'error': 'No active person'}), 404
    return jsonify(person.to_dict())


# ============== Photo APIs ==============

@api_bp.route('/folders', methods=['GET'])
def list_folders():
    """List all photo folders"""
    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    folders = scanner.get_folders()
    return jsonify(folders)


@api_bp.route('/photos', methods=['GET'])
def list_photos():
    """List photos, optionally filtered by folder"""
    folder = request.args.get('folder')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)

    query = Photo.query
    if folder:
        query = query.filter_by(folder_name=folder)

    query = query.order_by(Photo.file_name)
    pagination = query.paginate(page=page, per_page=per_page)

    return jsonify({
        'photos': [p.to_dict() for p in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'page': page
    })


@api_bp.route('/photos/index', methods=['POST'])
def index_photos():
    """Index photos from disk"""
    data = request.get_json() or {}
    folder = data.get('folder')

    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    count = scanner.index_photos(folder)

    return jsonify({'indexed': count})


@api_bp.route('/photos/<int:photo_id>', methods=['GET'])
def get_photo(photo_id):
    """Get a photo by ID with faces"""
    photo = Photo.query.get_or_404(photo_id)
    return jsonify(photo.to_dict(include_faces=True))


@api_bp.route('/photos/<int:photo_id>/detect', methods=['POST'])
def detect_faces_in_photo(photo_id):
    """Detect faces in a photo"""
    photo = Photo.query.get_or_404(photo_id)

    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    face_count = scanner.process_photo(photo, compute_embeddings=True)

    return jsonify({
        'face_count': face_count,
        'faces': [f.to_dict() for f in photo.detected_faces]
    })


@api_bp.route('/photos/<int:photo_id>/faces/manual', methods=['POST'])
def create_manual_face(photo_id):
    """Create a face entry with manual bounding box"""
    photo = Photo.query.get_or_404(photo_id)
    data = request.get_json()

    top = data.get('top')
    right = data.get('right')
    bottom = data.get('bottom')
    left = data.get('left')

    if None in (top, right, bottom, left):
        return jsonify({'error': 'Bounding box coordinates required'}), 400

    # Create face entry
    face = DetectedFace(
        photo_id=photo.id,
        bbox_top=top,
        bbox_right=right,
        bbox_bottom=bottom,
        bbox_left=left
    )

    # Compute embedding
    embedding_service = EmbeddingService()
    embedding = embedding_service.extract_embedding(
        photo.file_path,
        (top, right, bottom, left)
    )

    if embedding is not None:
        face.embedding = embedding

    db.session.add(face)
    photo.face_count = (photo.face_count or 0) + 1
    db.session.commit()

    return jsonify({'face': face.to_dict()}), 201


@api_bp.route('/photos/<int:photo_id>/image', methods=['GET'])
def get_photo_image(photo_id):
    """Get the actual photo image"""
    photo = Photo.query.get_or_404(photo_id)
    return send_file(photo.file_path)


@api_bp.route('/photos/<int:photo_id>/thumbnail', methods=['GET'])
def get_photo_thumbnail(photo_id):
    """Get or generate thumbnail for a photo"""
    photo = Photo.query.get_or_404(photo_id)
    size = request.args.get('size', 300, type=int)

    # Generate thumbnail path
    thumb_dir = current_app.config['THUMBNAILS_DIR']
    hash_name = hashlib.md5(photo.file_path.encode()).hexdigest()
    thumb_path = os.path.join(thumb_dir, f"{hash_name}_{size}.jpg")

    # Generate if doesn't exist
    if not os.path.exists(thumb_path):
        detector = FaceDetectionService()
        detector.create_thumbnail(photo.file_path, thumb_path, (size, size))

    return send_file(thumb_path)


# ============== Face APIs ==============

@api_bp.route('/faces/<int:face_id>', methods=['GET'])
def get_face(face_id):
    """Get a face by ID"""
    face = DetectedFace.query.get_or_404(face_id)
    return jsonify(face.to_dict())


@api_bp.route('/faces/<int:face_id>/adjust', methods=['PUT'])
def adjust_face_bbox(face_id):
    """Adjust face bounding box and recompute embedding"""
    face = DetectedFace.query.get_or_404(face_id)
    data = request.get_json()

    # Update bbox
    face.adjusted_top = data.get('top', face.bbox_top)
    face.adjusted_right = data.get('right', face.bbox_right)
    face.adjusted_bottom = data.get('bottom', face.bbox_bottom)
    face.adjusted_left = data.get('left', face.bbox_left)

    # Recompute embedding
    embedding_service = EmbeddingService()
    bbox = face.effective_bbox
    embedding = embedding_service.extract_embedding(face.photo.file_path, bbox)

    if embedding is not None:
        face.embedding = embedding

    db.session.commit()

    return jsonify(face.to_dict())


@api_bp.route('/faces/<int:face_id>/thumbnail', methods=['GET'])
def get_face_thumbnail(face_id):
    """Get thumbnail of a face region"""
    face = DetectedFace.query.get_or_404(face_id)
    padding = request.args.get('padding', 0.3, type=float)

    detector = FaceDetectionService()
    face_img = detector.extract_face_region(
        face.photo.file_path,
        face.effective_bbox,
        padding=padding
    )

    # Save to temp and return
    import io
    img_io = io.BytesIO()
    face_img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


# ============== Example Collection APIs ==============

@api_bp.route('/persons/<int:person_id>/examples', methods=['GET'])
def get_person_examples(person_id):
    """Get face examples for a person"""
    person = Person.query.get_or_404(person_id)
    is_test_set = request.args.get('test_set', 'false').lower() == 'true'

    examples = FaceExample.query.filter_by(
        person_id=person_id,
        is_test_set=is_test_set
    ).all()

    return jsonify([e.to_dict() for e in examples])


@api_bp.route('/persons/<int:person_id>/examples', methods=['POST'])
def add_face_example(person_id):
    """Add a face as training example"""
    person = Person.query.get_or_404(person_id)
    data = request.get_json()

    face_id = data.get('face_id')
    is_test_set = data.get('is_test_set', False)
    is_positive = data.get('is_positive', True)

    face = DetectedFace.query.get_or_404(face_id)

    # Check if already added
    existing = FaceExample.query.filter_by(
        person_id=person_id,
        detected_face_id=face_id
    ).first()

    if existing:
        return jsonify({'error': 'Face already added as example'}), 400

    # Ensure face has embedding
    if face.embedding is None:
        embedding_service = EmbeddingService()
        embedding = embedding_service.extract_embedding(face.photo.file_path, face.effective_bbox)
        if embedding is not None:
            face.embedding = embedding
        else:
            return jsonify({'error': 'Could not extract embedding for this face'}), 400

    example = FaceExample(
        person_id=person_id,
        detected_face_id=face_id,
        is_test_set=is_test_set,
        is_positive=is_positive
    )

    db.session.add(example)
    db.session.commit()

    return jsonify(example.to_dict()), 201


@api_bp.route('/persons/<int:person_id>/examples/<int:example_id>', methods=['DELETE'])
def remove_face_example(person_id, example_id):
    """Remove a face example"""
    example = FaceExample.query.filter_by(
        id=example_id,
        person_id=person_id
    ).first_or_404()

    db.session.delete(example)
    db.session.commit()

    return jsonify({'success': True})


@api_bp.route('/persons/<int:person_id>/diversity', methods=['GET'])
def get_diversity_analysis(person_id):
    """Get diversity analysis for collected examples"""
    person = Person.query.get_or_404(person_id)

    # Get training examples (not test set)
    examples = FaceExample.query.filter_by(
        person_id=person_id,
        is_test_set=False,
        is_positive=True
    ).all()

    embeddings = [e.detected_face.embedding for e in examples if e.detected_face.embedding is not None]

    analyzer = DiversityAnalyzer(embeddings)
    analysis = analyzer.analyze()

    return jsonify(analysis)


# ============== Training APIs ==============

@api_bp.route('/persons/<int:person_id>/train', methods=['POST'])
def train_classifier(person_id):
    """Train a classifier for a person"""
    person = Person.query.get_or_404(person_id)
    data = request.get_json() or {}
    algorithm = data.get('algorithm', 'svm')
    force_retrain = data.get('force_retrain', False)

    # Check if we already have a classifier with this algorithm
    existing = Classifier.query.filter_by(
        person_id=person_id,
        algorithm=algorithm
    ).first()

    if existing and not force_retrain:
        # Just activate the existing classifier and return it
        Classifier.query.filter_by(person_id=person_id).update({'is_active': False})
        existing.is_active = True
        db.session.commit()
        return jsonify(existing.to_dict())

    # Get positive examples
    positive_examples = FaceExample.query.filter_by(
        person_id=person_id,
        is_test_set=False,
        is_positive=True
    ).all()

    if len(positive_examples) < 3:
        return jsonify({'error': 'Need at least 3 positive examples'}), 400

    # For MobileFaceNet, extract embeddings using MobileFaceNet service
    if algorithm == 'mobilefacenet':
        from app.services.mobilefacenet_service import MobileFaceNetService
        mfn_service = MobileFaceNetService()

        positive_embeddings = []
        for example in positive_examples:
            face = example.detected_face
            embedding = mfn_service.extract_embedding(
                face.photo.file_path,
                face.effective_bbox
            )
            if embedding is not None:
                positive_embeddings.append(embedding)

        # Get negative examples using MobileFaceNet embeddings
        negative_embeddings = []
        for example in positive_examples:
            photo = example.detected_face.photo
            for face in photo.detected_faces:
                if face.id != example.detected_face_id:
                    is_positive = FaceExample.query.filter_by(
                        person_id=person_id,
                        detected_face_id=face.id,
                        is_positive=True
                    ).first()
                    if not is_positive:
                        embedding = mfn_service.extract_embedding(
                            face.photo.file_path,
                            face.effective_bbox
                        )
                        if embedding is not None:
                            negative_embeddings.append(embedding)
    else:
        # Use stored OpenFace embeddings for other algorithms
        positive_embeddings = [e.detected_face.embedding for e in positive_examples
                               if e.detected_face.embedding is not None]

        # Get negative examples (faces from same photos but not marked as positive)
        negative_embeddings = []
        for example in positive_examples:
            photo = example.detected_face.photo
            for face in photo.detected_faces:
                if face.id != example.detected_face_id and face.embedding is not None:
                    # Check if this face is not marked as positive for this person
                    is_positive = FaceExample.query.filter_by(
                        person_id=person_id,
                        detected_face_id=face.id,
                        is_positive=True
                    ).first()
                    if not is_positive:
                        negative_embeddings.append(face.embedding)

    # If not enough negatives, we can still train centroid/knn/mobilefacenet
    if algorithm == 'svm' and len(negative_embeddings) < 1:
        return jsonify({'error': 'Need negative examples for SVM. Use centroid or knn algorithm.'}), 400

    # Train and save
    try:
        clf, model_path, training_info = train_and_save_classifier(
            person_id, algorithm,
            positive_embeddings, negative_embeddings,
            current_app.config['MODELS_DIR']
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Deactivate other classifiers for this person
    Classifier.query.filter_by(person_id=person_id).update({'is_active': False})

    if existing:
        # Update existing classifier
        existing.model_path = model_path
        existing.num_training_examples = len(positive_embeddings)
        existing.num_negative_examples = len(negative_embeddings)
        existing.is_active = True
        existing.created_at = datetime.utcnow()
        db.session.commit()
        result = existing.to_dict()
    else:
        # Create new classifier
        classifier = Classifier(
            person_id=person_id,
            algorithm=algorithm,
            model_path=model_path,
            num_training_examples=len(positive_embeddings),
            num_negative_examples=len(negative_embeddings),
            is_active=True
        )
        db.session.add(classifier)
        db.session.commit()
        result = classifier.to_dict()

    # Add training info to result
    result['training_info'] = training_info
    if training_info.get('quality_warning'):
        result['quality_warning'] = training_info['quality_warning']

    return jsonify(result), 201 if not existing else 200


@api_bp.route('/classifiers/<int:classifier_id>/evaluate', methods=['POST'])
def evaluate_classifier(classifier_id):
    """Evaluate classifier on test set"""
    classifier_model = Classifier.query.get_or_404(classifier_id)

    # Load classifier
    clf = load_classifier(classifier_model)

    # Get test examples
    test_examples = FaceExample.query.filter_by(
        person_id=classifier_model.person_id,
        is_test_set=True
    ).all()

    if not test_examples:
        return jsonify({'error': 'No test examples found'}), 400

    embeddings = [e.detected_face.embedding for e in test_examples
                  if e.detected_face.embedding is not None]
    labels = [e.is_positive for e in test_examples
              if e.detected_face.embedding is not None]

    # Evaluate
    metrics = clf.evaluate(embeddings, labels)

    # Update classifier metrics
    classifier_model.test_accuracy = metrics['accuracy']
    classifier_model.test_precision = metrics['precision']
    classifier_model.test_recall = metrics['recall']
    classifier_model.test_f1 = metrics['f1']
    db.session.commit()

    return jsonify(metrics)


@api_bp.route('/classifiers/<int:classifier_id>/threshold', methods=['PUT'])
def update_threshold(classifier_id):
    """Update classifier threshold"""
    classifier_model = Classifier.query.get_or_404(classifier_id)
    data = request.get_json()

    threshold = data.get('threshold')
    if threshold is None:
        return jsonify({'error': 'Threshold is required'}), 400

    classifier_model.optimal_threshold = threshold

    # Re-evaluate with new threshold
    clf = load_classifier(classifier_model)
    clf.set_threshold(threshold)
    clf.save(classifier_model.model_path)

    db.session.commit()

    return jsonify(classifier_model.to_dict())


@api_bp.route('/classifiers/active', methods=['GET'])
def get_active_classifier():
    """Get the active classifier for the active person"""
    person = Person.query.filter_by(is_active=True).first()
    if not person:
        return jsonify({'error': 'No active person'}), 404

    classifier = Classifier.query.filter_by(
        person_id=person.id,
        is_active=True
    ).first()

    if not classifier:
        return jsonify({'error': 'No active classifier'}), 404

    return jsonify(classifier.to_dict())


# ============== Scan APIs ==============

@api_bp.route('/scan', methods=['POST'])
def start_scan():
    """Start scanning photos with the active classifier"""
    data = request.get_json() or {}
    classifier_id = data.get('classifier_id')
    folder = data.get('folder')

    if classifier_id:
        classifier = Classifier.query.get_or_404(classifier_id)
    else:
        person = Person.query.filter_by(is_active=True).first()
        if not person:
            return jsonify({'error': 'No active person'}), 400
        classifier = Classifier.query.filter_by(person_id=person.id, is_active=True).first()
        if not classifier:
            return jsonify({'error': 'No active classifier'}), 400

    # Clear previous results for this classifier
    ScanResult.query.filter_by(classifier_id=classifier.id).delete()
    db.session.commit()

    # Run scan
    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    results = list(scanner.scan_all_photos(classifier, folder))

    total_photos = len(results)
    photos_with_matches = sum(1 for r in results if r['has_match'])

    return jsonify({
        'total_photos': total_photos,
        'photos_with_matches': photos_with_matches,
        'classifier_id': classifier.id
    })


@api_bp.route('/scan/results', methods=['GET'])
def get_scan_results():
    """Get scan results"""
    classifier_id = request.args.get('classifier_id', type=int)
    folder = request.args.get('folder')
    matches_only = request.args.get('matches_only', 'false').lower() == 'true'

    if not classifier_id:
        person = Person.query.filter_by(is_active=True).first()
        if person:
            classifier = Classifier.query.filter_by(person_id=person.id, is_active=True).first()
            if classifier:
                classifier_id = classifier.id

    if not classifier_id:
        return jsonify({'error': 'No classifier specified'}), 400

    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    results = scanner.get_scan_results(classifier_id, folder, matches_only)

    # Group by folder
    by_folder = {}
    for result in results:
        folder_name = result['photo']['folder_name']
        if folder_name not in by_folder:
            by_folder[folder_name] = []
        by_folder[folder_name].append(result)

    return jsonify({
        'results': results,
        'by_folder': by_folder,
        'total': len(results),
        'with_matches': sum(1 for r in results if r['match_count'] > 0)
    })


@api_bp.route('/scan/results/<int:result_id>/feedback', methods=['PUT'])
def submit_feedback(result_id):
    """Submit user feedback on a scan result"""
    result = ScanResult.query.get_or_404(result_id)
    data = request.get_json()

    is_correct = data.get('is_correct')
    if is_correct is None:
        return jsonify({'error': 'is_correct is required'}), 400

    result.user_verified = is_correct
    result.verified_at = datetime.utcnow()
    db.session.commit()

    return jsonify(result.to_dict())
