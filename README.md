# FaceID

A face identification application for finding a specific person across a photo library. Built with Python, Flask, and OpenCV.

## Features

- **Phase 1: Face Collection** - Select and annotate face examples of a target person
- **Phase 2: Training & Evaluation** - Train a classifier and evaluate accuracy on test photos
- **Phase 3: Full Scan** - Scan entire photo library and view results in a gallery

## Tech Stack

- **Backend**: Python 3.10+, Flask, SQLAlchemy
- **Face Detection**: OpenCV DNN (SSD ResNet)
- **Face Embeddings**: OpenFace (128D vectors)
- **Classifier**: Centroid-based distance matching
- **Database**: SQLite

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/faceID.git
cd faceID

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data/models
```

## Usage

```bash
# Run with photos directory argument
python run.py /path/to/photos

# Or set environment variable
PHOTOS_DIR=/path/to/photos python run.py

# Custom port
python run.py /path/to/photos --port 8080
```

Then open http://127.0.0.1:5001 in your browser.

## Workflow

### Phase 1: Collect Face Examples
1. Browse your photo folders
2. Click on photos containing your target person
3. Draw a bounding box around their face
4. Collect ~10 diverse examples (different angles, lighting)

### Phase 2: Train & Evaluate
1. Select ~20 test photos containing your target person
2. Click "I'm Done - Train Classifier"
3. Review accuracy metrics
4. Adjust threshold if needed

### Phase 3: Scan All Photos
1. Click "Start Scan" to process all photos
2. View the gallery with matches highlighted in green
3. Summary shows total photos and match count

## Project Structure

```
faceID/
├── app/
│   ├── models/          # Database models & classifiers
│   ├── routes/          # Flask blueprints (API, phases)
│   ├── services/        # Face detection, embeddings, scanning
│   ├── static/          # CSS, JS, thumbnails
│   └── templates/       # Jinja2 HTML templates
├── data/
│   ├── faceID.db        # SQLite database
│   └── models/          # Downloaded ML models
├── run.py               # Entry point
└── requirements.txt
```

## API Endpoints

- `GET /api/photos` - List photos
- `GET /api/folders` - List photo folders
- `POST /api/photos/<id>/detect` - Detect faces in photo
- `POST /api/persons/<id>/train` - Train classifier
- `POST /api/scan` - Start full photo scan
- `GET /api/scan/results` - Get scan results

## License

MIT
