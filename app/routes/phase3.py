from flask import Blueprint, render_template, current_app
from app.models.database import Person, Classifier, ScanResult
from app.services.photo_scanner import PhotoScanner

phase3_bp = Blueprint('phase3', __name__, url_prefix='/phase3')


@phase3_bp.route('/')
def index():
    """Phase 3 dashboard - full scan"""
    person = Person.query.filter_by(is_active=True).first()

    # Get folders
    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    folders = scanner.get_folders()

    classifier = None
    scan_results_count = 0

    if person:
        classifier = Classifier.query.filter_by(
            person_id=person.id,
            is_active=True
        ).first()

        if classifier:
            scan_results_count = ScanResult.query.filter_by(
                classifier_id=classifier.id
            ).count()

    return render_template('phase3/index.html',
                           person=person,
                           folders=folders,
                           classifier=classifier,
                           scan_results_count=scan_results_count)


@phase3_bp.route('/gallery')
def gallery():
    """View scan results gallery"""
    person = Person.query.filter_by(is_active=True).first()

    # Get folders
    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    folders = scanner.get_folders()

    classifier = None
    if person:
        classifier = Classifier.query.filter_by(
            person_id=person.id,
            is_active=True
        ).first()

    return render_template('phase3/gallery.html',
                           person=person,
                           folders=folders,
                           classifier=classifier)
