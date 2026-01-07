from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
import logging

db = SQLAlchemy()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('faceID')

def create_app(photos_dir=None):
    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'faceID.db')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['PHOTOS_DIR'] = photos_dir or os.environ.get('PHOTOS_DIR', '/Users/jx/dev/photos')
    app.config['THUMBNAILS_DIR'] = os.path.join(os.path.dirname(__file__), 'static', 'thumbnails')
    app.config['MODELS_DIR'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from app.routes.phase1 import phase1_bp
    from app.routes.phase2 import phase2_bp
    from app.routes.phase3 import phase3_bp
    from app.routes.api import api_bp

    app.register_blueprint(phase1_bp)
    app.register_blueprint(phase2_bp)
    app.register_blueprint(phase3_bp)
    app.register_blueprint(api_bp)

    # Create tables
    with app.app_context():
        db.create_all()

    return app
