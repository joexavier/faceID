from flask import Blueprint, render_template, current_app
from app.models.database import Person, Photo, FaceExample
from app.services.photo_scanner import PhotoScanner

phase1_bp = Blueprint('phase1', __name__, url_prefix='/phase1')


@phase1_bp.route('/')
def index():
    """Phase 1 dashboard - example collection"""
    # Get or create active person
    person = Person.query.filter_by(is_active=True).first()

    # Get folders
    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    folders = scanner.get_folders()

    # Index photos if not done
    scanner.index_photos()

    # Get example count
    example_count = 0
    if person:
        example_count = FaceExample.query.filter_by(
            person_id=person.id,
            is_test_set=False,
            is_positive=True
        ).count()

    return render_template('phase1/index.html',
                           person=person,
                           folders=folders,
                           example_count=example_count)


@phase1_bp.route('/browse')
def browse():
    """Browse photos to select faces"""
    person = Person.query.filter_by(is_active=True).first()

    scanner = PhotoScanner(current_app.config['PHOTOS_DIR'])
    folders = scanner.get_folders()

    return render_template('phase1/browse.html',
                           person=person,
                           folders=folders)


@phase1_bp.route('/photo/<int:photo_id>')
def collect(photo_id):
    """Collect face examples from a specific photo"""
    person = Person.query.filter_by(is_active=True).first()
    photo = Photo.query.get_or_404(photo_id)

    return render_template('phase1/collect.html',
                           person=person,
                           photo=photo)
