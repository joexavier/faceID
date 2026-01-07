#!/usr/bin/env python3
"""
FaceID Application Entry Point
Run with: python run.py /path/to/photos
"""

import argparse
import os
from app import create_app

parser = argparse.ArgumentParser(description='FaceID - Face identification app')
parser.add_argument('photos_dir', nargs='?', default=None,
                    help='Path to photos directory (default: /Users/jx/dev/photos or PHOTOS_DIR env var)')
parser.add_argument('--port', type=int, default=5001, help='Port to run on (default: 5001)')
args = parser.parse_args()

# Validate photos directory
photos_dir = args.photos_dir
if photos_dir:
    if not os.path.isdir(photos_dir):
        print(f"Error: '{photos_dir}' is not a valid directory")
        exit(1)
    photos_dir = os.path.abspath(photos_dir)

app = create_app(photos_dir=photos_dir)


@app.route('/')
def home():
    """Redirect to Phase 1"""
    from flask import redirect, url_for
    return redirect(url_for('phase1.index'))


if __name__ == '__main__':
    print("Starting FaceID application...")
    print(f"Photos directory: {app.config['PHOTOS_DIR']}")
    print(f"Open http://127.0.0.1:{args.port} in your browser")
    app.run(debug=True, host='127.0.0.1', port=args.port)
