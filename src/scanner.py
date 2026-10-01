import os
import re
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import exifread
except ImportError:
    exifread = None

SUPPORTED_PHOTOS = {
    '.jpg', '.jpeg', '.png', '.webp', '.heic', 
    '.heif', '.tiff', '.tif', '.gif', '.insp', '.bmp'
}

SUPPORTED_VIDEOS = {
    '.mp4', '.mov', '.avi', '.m4v', '.3gp', 
    '.mpeg', '.mpg', '.ts', '.webm', '.mkv', '.insv'
}

EXIFTOOL_PATH = shutil.which("exiftool")


def parse_date_and_timestamp(raw_date: Any, fallback_mtime: float = 0.0) -> Tuple[str, float]:
    """Parses EXIF/QuickTime date strings into display string and epoch timestamp."""
    if not raw_date:
        dt = datetime.fromtimestamp(fallback_mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S"), fallback_mtime

    s = str(raw_date).strip()

    if len(s) >= 10 and s[4] == ':' and s[7] == ':':
        s = s[:4] + '-' + s[5:7] + '-' + s[8:]

    s_iso = s.replace(" ", "T", 1) if " " in s else s

    try:
        dt = datetime.fromisoformat(s_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.timestamp()
    except Exception:
        pass

    m = re.match(r"(\d{4})[-:](\d{2})[-:](\d{2})[T ](\d{2}):(\d{2}):(\d{2})", s)
    if m:
        try:
            dt = datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]), tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.timestamp()
        except Exception:
            pass

    dt = datetime.fromtimestamp(fallback_mtime, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S"), fallback_mtime


def _convert_dms_to_degrees(value) -> Optional[float]:
    """Convert EXIF GPS DMS tuple/Ratio list to Decimal Degrees."""
    try:
        vals = value.values if hasattr(value, 'values') else value
        if len(vals) < 3:
            return None
        
        def to_float(x):
            if hasattr(x, 'num') and hasattr(x, 'den'):
                return float(x.num) / float(x.den) if x.den != 0 else 0.0
            return float(x)

        d = to_float(vals[0])
        m = to_float(vals[1])
        s = to_float(vals[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def _parse_gps_coord(item: dict) -> Tuple[Optional[float], Optional[float]]:
    """Extracts signed float latitude and longitude across photos and videos."""
    comp_lat = item.get("Composite:GPSLatitude") or item.get("GPSLatitude")
    comp_lon = item.get("Composite:GPSLongitude") or item.get("GPSLongitude")
    gps_pos = (
        item.get("Composite:GPSPosition") or item.get("GPSPosition") or 
        item.get("GPSCoordinates") or item.get("Keys:GPSCoordinates") or 
        item.get("UserData:GPSCoordinates")
    )

    def clean_val(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            s = str(val).strip()
            m = re.search(r"([-+]?\d+\.?\d*)", s)
            if m:
                res = float(m.group(1))
                if 'S' in s.upper() or 'W' in s.upper():
                    res = -abs(res)
                return res
        return None

    lat = clean_val(comp_lat)
    lon = clean_val(comp_lon)

    lat_ref = str(item.get("GPSLatitudeRef") or item.get("EXIF:GPSLatitudeRef") or "").upper()
    lon_ref = str(item.get("GPSLongitudeRef") or item.get("EXIF:GPSLongitudeRef") or "").upper()

    if lat is not None and lat_ref:
        if 'S' in lat_ref:
            lat = -abs(lat)
        elif 'N' in lat_ref:
            lat = abs(lat)

    if lon is not None and lon_ref:
        if 'W' in lon_ref:
            lon = -abs(lon)
        elif 'E' in lon_ref:
            lon = abs(lon)

    if (lat is None or lon is None) and gps_pos:
        try:
            parts = str(gps_pos).split(",")
            if len(parts) >= 2:
                p_lat = clean_val(parts[0])
                p_lon = clean_val(parts[1])
                if lat is None: lat = p_lat
                if lon is None: lon = p_lon
        except Exception:
            pass

    return lat, lon


def _get_metadata_exiftool_batch(filepaths: List[str]) -> Dict[str, dict]:
    """Extracts metadata from a batch of files using ExifTool with signed GPS calculation."""
    results = {}
    if not EXIFTOOL_PATH or not filepaths:
        return results

    cmd = [
        EXIFTOOL_PATH,
        "-json",
        "-n",                   # Signed decimal coordinates
        "-q",                   # Quiet mode
        "-m",                   # Ignore minor warnings
        "-api", "QuickTimeUTC", # Proper timezone handling for MP4/MOV
        "-charset", "filename=utf8",
        "-DateTimeOriginal",
        "-CreateDate",
        "-MediaCreateDate",
        "-TrackCreateDate",
        "-ModifyDate",
        "-Model",
        "-AndroidModel",
        "-Make",
        "-AndroidMake",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSLatitudeRef",
        "-GPSLongitudeRef",
        "-GPSPosition",
        "-Composite:GPSLatitude",
        "-Composite:GPSLongitude",
        "-Composite:GPSPosition",
        "-Keys:GPSCoordinates",
        "-UserData:GPSCoordinates",
        "-Duration",
        "--",
        *filepaths
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        stdout = proc.stdout.strip() if proc.stdout else ""
        idx = stdout.find("[")
        if idx != -1:
            raw_items = json.loads(stdout[idx:])

            for item in raw_items:
                source = item.get("SourceFile")
                if not source:
                    continue

                mtime = 0.0
                try:
                    mtime = os.path.getmtime(source)
                except OSError:
                    pass

                raw_date = (
                    item.get("DateTimeOriginal")
                    or item.get("CreateDate")
                    or item.get("MediaCreateDate")
                    or item.get("TrackCreateDate")
                    or item.get("ModifyDate")
                )
                date_str, date_ts = parse_date_and_timestamp(raw_date, mtime)

                model = str(item.get("AndroidModel") or item.get("Model") or "").strip()
                make = str(item.get("AndroidMake") or item.get("Make") or "").strip()
                camera = "Unknown"
                if model and make:
                    camera = model if make.lower() in model.lower() else f"{make} {model}"
                elif model:
                    camera = model
                elif make:
                    camera = make

                lat, lon = _parse_gps_coord(item)

                meta_entry = {
                    'date': date_str,
                    'date_ts': date_ts,
                    'camera': camera.strip() or "Unknown",
                    'lat': lat,
                    'lon': lon
                }

                results[source] = meta_entry
                results[os.path.normpath(source)] = meta_entry
                results[os.path.abspath(source)] = meta_entry
    except Exception as e:
        print(f"ExifTool batch error: {e}")

    return results


def _get_metadata_fallback(filepath: str) -> dict:
    """Python-based fallback for metadata extraction when ExifTool is unavailable."""
    mtime = 0.0
    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        pass

    metadata = {
        'date': datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'date_ts': mtime,
        'camera': 'Unknown',
        'lat': None,
        'lon': None
    }

    if not exifread or Path(filepath).suffix.lower() in SUPPORTED_VIDEOS:
        return metadata

    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)

            raw_date = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
            if raw_date:
                d_str, d_ts = parse_date_and_timestamp(str(raw_date), mtime)
                metadata['date'] = d_str
                metadata['date_ts'] = d_ts

            model = str(tags.get('Image Model', '')).strip()
            make = str(tags.get('Image Make', '')).strip()
            if model and make:
                metadata['camera'] = model if make.lower() in model.lower() else f"{make} {model}"
            elif model:
                metadata['camera'] = model

            lat_tag = tags.get('GPS GPSLatitude')
            lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N')).upper()
            lon_tag = tags.get('GPS GPSLongitude')
            lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E')).upper()

            if lat_tag:
                lat = _convert_dms_to_degrees(lat_tag)
                if lat is not None:
                    if 'S' in lat_ref:
                        lat = -abs(lat)
                    metadata['lat'] = lat

            if lon_tag:
                lon = _convert_dms_to_degrees(lon_tag)
                if lon is not None:
                    if 'W' in lon_ref:
                        lon = -abs(lon)
                    metadata['lon'] = lon
    except Exception:
        pass

    return metadata


def get_metadata_batch(filepaths: List[str], chunk_size: int = 500) -> Dict[str, dict]:
    """Extracts metadata in batches of 500 using ExifTool."""
    results = {}
    if not filepaths:
        return results

    if EXIFTOOL_PATH:
        for i in range(0, len(filepaths), chunk_size):
            chunk = filepaths[i:i + chunk_size]
            chunk_results = _get_metadata_exiftool_batch(chunk)
            results.update(chunk_results)

    for fp in filepaths:
        norm = os.path.normpath(fp)
        if fp not in results and norm not in results:
            results[fp] = _get_metadata_fallback(fp)

    return results


def scan_directory(base_paths: list):
    """Recursively yields media file information."""
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue

        for root, _, files in os.walk(base_path):
            for file in files:
                filepath = os.path.join(root, file)
                ext = Path(file).suffix.lower()
                
                media_type = None
                if ext in SUPPORTED_PHOTOS:
                    media_type = "360_photo" if ext == '.insp' else "photo"
                elif ext in SUPPORTED_VIDEOS:
                    media_type = "video"

                if media_type:
                    try:
                        stat = os.stat(filepath)
                        yield {
                            'path': filepath,
                            'type': media_type,
                            'size': stat.st_size,
                            'mtime': stat.st_mtime
                        }
                    except OSError:
                        continue
