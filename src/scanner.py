import os
import subprocess
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import exifread

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.tiff', '.tif', '.bmp', '.insp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.m4v', '.3gp', '.mpeg', '.mpg', '.ts', '.webm', '.mkv', '.insv'}


def scan_directory(directories: List[str]):
    """Recursively walks directories and yields files matching photo/video extensions."""
    for d in directories:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                full_path = os.path.join(root, file)
                
                # Determine media type
                media_type = None
                if ext in IMAGE_EXTS:
                    media_type = "photo"
                elif ext in VIDEO_EXTS:
                    media_type = "video"
                
                if media_type:
                    try:
                        mtime = os.path.getmtime(full_path)
                        yield {
                            "path": full_path,
                            "type": media_type,
                            "mtime": mtime
                        }
                    except Exception as e:
                        print(f"Error reading file stat for {full_path}: {e}")


def parse_gps_coord(coord, ref):
    if not coord or not ref:
        return None
    try:
        degrees = float(coord[0].num) / float(coord[0].den)
        minutes = float(coord[1].num) / float(coord[1].den)
        seconds = float(coord[2].num) / float(coord[2].den)
        val = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            val = -val
        return val
    except Exception:
        return None


def get_metadata(path: str) -> Dict[str, Any]:
    """Extracts date, camera model, and GPS coordinates using ExifRead or FFprobe fallback."""
    ext = os.path.splitext(path)[1].lower()
    
    # 1. Video Metadata via FFprobe (handles MP4, MOV, INSV, etc.)
    if ext in VIDEO_EXTS:
        return _get_video_metadata(path)

    # 2. Image Metadata via ExifRead
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # Date
            date_str = None
            date_ts = None
            for date_tag in ['EXIF DateTimeOriginal', 'Image DateTime', 'EXIF DateTimeDigitized']:
                if date_tag in tags:
                    try:
                        raw = str(tags[date_tag])
                        dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        date_ts = dt.timestamp()
                        break
                    except Exception:
                        pass

            # Camera
            camera = "Unknown"
            make = str(tags.get('Image Make', '')).strip()
            model = str(tags.get('Image Model', '')).strip()
            if make or model:
                if model and make and make.lower() in model.lower():
                    camera = model
                else:
                    camera = f"{make} {model}".strip()

            # GPS
            lat = parse_gps_coord(tags.get('GPS GPSLatitude'), str(tags.get('GPS GPSLatitudeRef', '')))
            lon = parse_gps_coord(tags.get('GPS GPSLongitude'), str(tags.get('GPS GPSLongitudeRef', '')))

            return {
                "date": date_str,
                "date_ts": date_ts,
                "camera": camera,
                "lat": lat,
                "lon": lon
            }
    except Exception:
        pass

    # Fallback to file modification time if EXIF parsing fails
    try:
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return {
            "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date_ts": mtime,
            "camera": "Unknown",
            "lat": None,
            "lon": None
        }
    except Exception:
        return {}


def _get_video_metadata(path: str) -> Dict[str, Any]:
    """Runs ffprobe to extract video creation date, camera tags, and GPS metadata."""
    ext = os.path.splitext(path)[1].lower()
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if proc.returncode == 0:
            data = json.loads(proc.stdout.decode('utf-8', errors='ignore'))
            format_tags = data.get("format", {}).get("tags", {})
            
            # Inspect first video stream tags
            stream_tags = {}
            streams = data.get("streams", [])
            for s in streams:
                if s.get("codec_type") == "video" and "tags" in s:
                    stream_tags = s["tags"]
                    break

            # Check QuickTime metadata keys used by Insta360
            camera = (
                format_tags.get("com.apple.quicktime.model") or
                format_tags.get("model") or
                format_tags.get("make") or
                stream_tags.get("model") or
                stream_tags.get("handler_name") or
                format_tags.get("encoder")
            )

            # Ensure proper camera branding for .insv / .insp
            if not camera or camera.lower() in ["unknown", "handler_name", "encoder", "video handler"]:
                camera = "Insta360" if ext in ['.insv', '.insp'] else "Unknown"
            elif ext in ['.insv', '.insp'] and "insta360" not in camera.lower():
                camera = f"Insta360 {camera}"

            # Extract date
            creation_time = (
                format_tags.get("com.apple.quicktime.creationdate") or 
                format_tags.get("creation_time") or 
                format_tags.get("date") or 
                stream_tags.get("creation_time")
            )
            date_str = None
            date_ts = None
            if creation_time:
                try:
                    clean_time = creation_time.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(clean_time).astimezone(timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    date_ts = dt.timestamp()
                except Exception:
                    pass

            if not date_ts:
                mtime = os.path.getmtime(path)
                dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                date_ts = mtime

            return {
                "date": date_str,
                "date_ts": date_ts,
                "camera": camera,
                "lat": None,
                "lon": None
            }
    except Exception as e:
        print(f"ffprobe metadata extraction failed for {path}: {e}")

    # Fallback if ffprobe fails
    try:
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return {
            "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date_ts": mtime,
            "camera": "Insta360" if ext in ['.insv', '.insp'] else "Unknown",
            "lat": None,
            "lon": None
        }
    except Exception:
        return {}


def get_metadata_batch(paths: List[str], chunk_size: int = 500) -> Dict[str, Dict[str, Any]]:
    """Extracts metadata for a list of file paths."""
    results = {}
    for path in paths:
        results[path] = get_metadata(path)
    return results
