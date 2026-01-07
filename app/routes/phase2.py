from flask import Blueprint, render_template
from app.models.database import Person, Classifier, FaceExample

phase2_bp = Blueprint('phase2', __name__, url_prefix='/phase2')


@phase2_bp.route('/')
def index():
    """Phase 2 dashboard - test set and training"""
    person = Person.query.filter_by(is_active=True).first()

    # Get counts
    training_count = 0
    test_count = 0
    classifier = None

    if person:
        training_count = FaceExample.query.filter_by(
            person_id=person.id,
            is_test_set=False,
            is_positive=True
        ).count()

        test_count = FaceExample.query.filter_by(
            person_id=person.id,
            is_test_set=True
        ).count()

        classifier = Classifier.query.filter_by(
            person_id=person.id,
            is_active=True
        ).first()

    return render_template('phase2/index.html',
                           person=person,
                           training_count=training_count,
                           test_count=test_count,
                           classifier=classifier)


@phase2_bp.route('/results')
def results():
    """View training results"""
    person = Person.query.filter_by(is_active=True).first()

    classifier = None
    if person:
        classifier = Classifier.query.filter_by(
            person_id=person.id,
            is_active=True
        ).first()

    return render_template('phase2/results.html',
                           person=person,
                           classifier=classifier)
