#!/usr/bin/env python3
"""Backup the FaceID database"""

import shutil
import os
from datetime import datetime

DB_PATH = '/Users/jx/dev/faceID/data/faceID.db'
BACKUP_DIR = '/Users/jx/dev/faceID/data/backups'

def backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_DIR, f'faceID_{timestamp}.db')
    shutil.copy2(DB_PATH, backup_path)
    print(f'Backup created: {backup_path}')

    # Keep only last 5 backups
    backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.db')])
    for old_backup in backups[:-5]:
        os.remove(os.path.join(BACKUP_DIR, old_backup))
        print(f'Removed old backup: {old_backup}')

if __name__ == '__main__':
    backup()
