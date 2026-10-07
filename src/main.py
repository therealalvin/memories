import os
import json
import math
import sqlite3
import aiosqlite
import mimetypes
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from scanner import scan_directory, get_metadata_batch
from ml import (
    get_image_embedding, 
    get_image_embeddings_batch, 
    get_text_embedding,
    generate_dynamic_thumbnail_bytes,
    generate_dynamic_full_image_bytes,
    is_model_loaded,
    get_model_and_processor,
)

# Register .insv and .insp MIME types for media streaming in browser
mimetypes.add_type("video/mp4", ".insv")
mimetypes.add_type("image/jpeg", ".insp")

app = FastAPI()
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

DB_PATH = "/data/memories.db"

sync_status = {
    "running": False,
    "phase": "idle",
    "current": 0,
    "total": 0,
    "percent": 0,
    "message": "Idle"
}


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA busy_timeout=30000;")
    c.execute("PRAGMA synchronous=NORMAL;")
    
    c.execute('''CREATE TABLE IF NOT EXISTS media 
                 (id INTEGER PRIMARY KEY, path TEXT UNIQUE, type TEXT, 
                  mtime REAL, date TEXT, date_ts REAL, camera TEXT, 
                  lat REAL, lon REAL, embedding TEXT)''')

    for col, col_type in [("date_ts", "REAL"), ("lat", "REAL"), ("lon", "REAL"), ("camera", "TEXT")]:
        try:
            c.execute(f"ALTER TABLE media ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    c.execute("CREATE INDEX IF NOT EXISTS idx_media_path ON media(path);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_media_date_ts ON media(date_ts);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media(type);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_media_camera ON media(camera);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_media_lat_lon ON media(lat, lon);")
            
    conn.commit()
    conn.close()


init_db()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("/app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


class FilterItem(BaseModel):
    id: str
    type: str             # 'text', 'similar', 'media_type', 'similar_date', 'similar_location', 'camera'
    value: str            
    days: float = 7.0             
    radius_miles: float = 5.0     
    threshold: Optional[float] = None
    label: str


class QueryRequest(BaseModel):
    filters: List[FilterItem]
    sort_by: str = "date_desc"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 100


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def background_sync(force: bool = False):
    global sync_status
    sync_status["running"] = True
    sync_status["phase"] = "scanning"
    sync_status["current"] = 0
    sync_status["total"] = 0
    sync_status["percent"] = 2
    sync_status["message"] = "Scanning media directories for photos and videos..."

    try:
        files = []
        for f in scan_directory(["/media/pictures", "/media/videos"]):
            files.append(f)
            if len(files) % 100 == 0:
                sync_status["current"] = len(files)
                sync_status["message"] = f"Discovered {len(files)} photos and videos..."

        total_scanned = len(files)
        sync_status["current"] = total_scanned
        sync_status["percent"] = 5
        sync_status["phase"] = "analyzing"
        sync_status["message"] = f"Discovered {total_scanned} files. Checking database index..."

        conn = sqlite3.connect(DB_PATH, timeout=60.0)
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA busy_timeout=60000;")
        c.execute("PRAGMA synchronous=NORMAL;")

        # Query existing paths, mtimes, embedding presence, and date_ts presence
        c.execute("SELECT path, mtime, embedding IS NOT NULL, date_ts IS NOT NULL FROM media")
        existing_records = {row[0]: (float(row[1] or 0.0), bool(row[2]), bool(row[3])) for row in c.fetchall()}

        files_to_update = []
        files_needing_embedding = []

        for f in files:
            path = f["path"]
            mtime = f["mtime"]
            rec = existing_records.get(path)

            if force or not rec or (mtime - rec[0] > 0.001) or not rec[2]:
                files_to_update.append(f)
            
            if force or not rec or (mtime - rec[0] > 0.001) or not rec[1]:
                files_needing_embedding.append(f)

        total_tasks = len(files_to_update) + len(files_needing_embedding)
        sync_status["total"] = total_tasks
        sync_status["current"] = 0

        if total_tasks == 0:
            sync_status["phase"] = "complete"
            sync_status["percent"] = 100
            sync_status["current"] = total_scanned
            sync_status["total"] = total_scanned
            sync_status["message"] = f"All {total_scanned} files are already fully indexed and up to date!"
            sync_status["running"] = False
            conn.close()
            return

        completed_work = 0

        # Step A: Metadata and GPS extraction
        if files_to_update:
            sync_status["phase"] = "metadata"
            meta_chunk_size = 500
            update_paths = [f["path"] for f in files_to_update]

            for i in range(0, len(files_to_update), meta_chunk_size):
                chunk_files = files_to_update[i:i + meta_chunk_size]
                chunk_paths = update_paths[i:i + meta_chunk_size]

                sync_status["message"] = f"Extracting metadata & GPS ({completed_work + len(chunk_files)}/{total_tasks})..."
                meta_map = get_metadata_batch(chunk_paths, chunk_size=meta_chunk_size)

                records = []
                for f in chunk_files:
                    p = f["path"]
                    meta = meta_map.get(p, {}) or meta_map.get(os.path.normpath(p), {})
                    date_str = meta.get("date") or datetime.fromtimestamp(f["mtime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    date_ts = meta.get("date_ts", f["mtime"])
                    camera = meta.get("camera", "Unknown")
                    lat = meta.get("lat")
                    lon = meta.get("lon")
                    records.append((p, f["type"], f["mtime"], date_str, date_ts, camera, lat, lon))

                c.executemany('''
                    INSERT INTO media (path, type, mtime, date, date_ts, camera, lat, lon)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        type=excluded.type,
                        mtime=excluded.mtime,
                        date=excluded.date,
                        date_ts=excluded.date_ts,
                        camera=excluded.camera,
                        lat=excluded.lat,
                        lon=excluded.lon
                ''', records)
                conn.commit()

                completed_work += len(chunk_files)
                sync_status["current"] = completed_work
                sync_status["percent"] = max(5, int((completed_work / total_tasks) * 100))

        # Step B: GPU CLIP embedding extraction
        if files_needing_embedding:
            if not is_model_loaded():
                sync_status["phase"] = "embeddings"
                sync_status["message"] = "Initializing CLIP AI model on GPU..."
                get_model_and_processor()

            sync_status["phase"] = "embeddings"
            batch_size = 32
            embed_paths = [f["path"] for f in files_needing_embedding]
            total_embed = len(embed_paths)

            for i in range(0, total_embed, batch_size):
                chunk = embed_paths[i:i + batch_size]
                sync_status["message"] = f"Generating GPU embeddings ({completed_work + len(chunk)}/{total_tasks})..."

                embeddings = get_image_embeddings_batch(chunk, batch_size=batch_size)

                updates = []
                for path, emb in zip(chunk, embeddings):
                    if emb:
                        updates.append((json.dumps(emb), path))

                if updates:
                    c.executemany("UPDATE media SET embedding = ? WHERE path = ?", updates)
                    conn.commit()

                completed_work += len(chunk)
                sync_status["current"] = completed_work
                sync_status["percent"] = max(5, int((completed_work / total_tasks) * 100))

        # Step C: Clean up missing files
        sync_status["phase"] = "cleanup"
        sync_status["message"] = "Cleaning up removed files..."
        c.execute("SELECT path FROM media")
        db_paths = [row[0] for row in c.fetchall()]
        to_delete = [p for p in db_paths if not os.path.exists(p)]
        if to_delete:
            c.executemany("DELETE FROM media WHERE path = ?", [(p,) for p in to_delete])
            conn.commit()

        conn.close()
        sync_status["phase"] = "complete"
        sync_status["current"] = total_tasks
        sync_status["percent"] = 100
        sync_status["message"] = f"Sync complete! Processed {total_tasks} media items."
    except Exception as e:
        print(f"Background sync error: {e}")
        sync_status["phase"] = "error"
        sync_status["message"] = f"Sync error: {str(e)}"
    finally:
        sync_status["running"] = False


@app.post("/api/sync")
async def trigger_sync(background_tasks: BackgroundTasks, force: bool = False):
    if not sync_status["running"]:
        background_tasks.add_task(background_sync, force=force)
        return {"status": "Sync started", "force": force}
    return {"status": "Sync already running"}


@app.get("/api/sync/status")
async def get_sync_status():
    return sync_status


@app.get("/api/cameras")
async def get_cameras():
    async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=30000;")
        async with db.execute("SELECT DISTINCT camera FROM media WHERE camera IS NOT NULL AND camera != '' AND camera != 'Unknown' ORDER BY camera") as cursor:
            rows = await cursor.fetchall()
            return [r[0] for r in rows]


@app.get("/api/media_details/{media_id}")
async def media_details(media_id: int):
    async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=30000;")
        async with db.execute("SELECT id, path, type, date, date_ts, camera, lat, lon FROM media WHERE id = ?", (media_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                lat = float(row[6]) if row[6] is not None else None
                lon = float(row[7]) if row[7] is not None else None
                return {
                    "id": row[0],
                    "path": row[1],
                    "filename": os.path.basename(row[1]),
                    "type": row[2],
                    "date": row[3] or "Unknown",
                    "date_ts": float(row[4]) if row[4] is not None else 0.0,
                    "camera": row[5] or "Unknown",
                    "lat": lat,
                    "lon": lon
                }
            return {"error": "Not found"}


@app.get("/api/thumbnail/{media_id}")
async def get_thumbnail(media_id: int):
    """Generates and serves dynamic JPEG thumbnails on-the-fly directly from memory."""
    async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
        async with db.execute("SELECT path, type FROM media WHERE id = ?", (media_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                path, m_type = row[0], row[1]
                if os.path.exists(path):
                    thumb_bytes = generate_dynamic_thumbnail_bytes(path, m_type, max_size=360)
                    if thumb_bytes:
                        return Response(content=thumb_bytes, media_type="image/jpeg")
                    return FileResponse(path)
            return Response(status_code=404)


@app.post("/api/search")
async def search_cumulative(req: QueryRequest):
    """Cumulative multi-criteria search (Strict AND logic)."""
    async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=30000;")

        where_clauses = ["1=1"]
        params: List[Any] = []

        # Start Date and End Date preset bounds
        if req.start_date:
            try:
                dt_start = datetime.strptime(req.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                where_clauses.append("date_ts >= ?")
                params.append(dt_start.timestamp())
            except Exception as e:
                print(f"Error parsing start_date: {e}")

        if req.end_date:
            try:
                dt_end = datetime.strptime(req.end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                where_clauses.append("date_ts <= ?")
                params.append(dt_end.timestamp())
            except Exception as e:
                print(f"Error parsing end_date: {e}")

        # 1. Media format filter
        media_type = "all"
        for f in req.filters:
            if f.type == "media_type":
                media_type = f.value

        if media_type == "photos":
            where_clauses.append("type IN ('photo', '360_photo')")
        elif media_type == "videos":
            where_clauses.append("type = 'video'")

        # 2. Camera filter
        for f in req.filters:
            if f.type == "camera" and f.value.strip():
                where_clauses.append("LOWER(TRIM(camera)) LIKE LOWER(?)")
                params.append(f"%{f.value.strip()}%")

        # 3. Cumulative Date Windows
        date_filters = [f for f in req.filters if f.type == "similar_date"]
        if date_filters:
            min_bound = -1e12
            max_bound = 1e12
            has_valid_date_range = True

            for f in date_filters:
                try:
                    target_id = int(f.value)
                    async with db.execute("SELECT date_ts, mtime FROM media WHERE id = ?", (target_id,)) as cursor:
                        t_row = await cursor.fetchone()
                    if t_row and (t_row[0] is not None or t_row[1] is not None):
                        target_ts = float(t_row[0] if t_row[0] is not None else t_row[1])
                        tolerance_sec = float(f.days) * 86400.0
                        cur_min = target_ts - tolerance_sec
                        cur_max = target_ts + tolerance_sec
                        min_bound = max(min_bound, cur_min)
                        max_bound = min(max_bound, cur_max)
                    else:
                        has_valid_date_range = False
                except Exception:
                    has_valid_date_range = False

            if not has_valid_date_range or min_bound > max_bound:
                return []

            where_clauses.append("date_ts >= ? AND date_ts <= ?")
            params.extend([min_bound, max_bound])

        # 4. Location Radius (Bounding box optimization: supports ID lookup or "lat,lon" strings)
        location_filters = []
        for f in req.filters:
            if f.type == "similar_location":
                try:
                    if "," in str(f.value):
                        lat_str, lon_str = str(f.value).split(",")
                        location_filters.append((float(lat_str), float(lon_str), float(f.radius_miles)))
                    else:
                        target_id = int(f.value)
                        async with db.execute("SELECT lat, lon FROM media WHERE id = ?", (target_id,)) as cursor:
                            t_row = await cursor.fetchone()
                        if t_row and t_row[0] is not None and t_row[1] is not None:
                            t_lat, t_lon = float(t_row[0]), float(t_row[1])
                            location_filters.append((t_lat, t_lon, float(f.radius_miles)))
                except Exception as e:
                    print(f"Error parsing location filter: {e}")

        if location_filters:
            where_clauses.append("lat IS NOT NULL AND lon IS NOT NULL")
            overall_min_lat = -90.0
            overall_max_lat = 90.0
            overall_min_lon = -180.0
            overall_max_lon = 180.0

            for t_lat, t_lon, rad in location_filters:
                d_lat = rad / 69.0
                cos_lat = math.cos(math.radians(t_lat))
                d_lon = rad / (69.0 * cos_lat) if abs(cos_lat) > 0.001 else 180.0

                overall_min_lat = max(overall_min_lat, t_lat - d_lat)
                overall_max_lat = min(overall_max_lat, t_lat + d_lat)
                overall_min_lon = max(overall_min_lon, t_lon - d_lon)
                overall_max_lon = min(overall_max_lon, t_lon + d_lon)

            where_clauses.append("lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?")
            params.extend([overall_min_lat, overall_max_lat, overall_min_lon, overall_max_lon])

        # Vector & Text similarity filters
        vector_filters = [f for f in req.filters if f.type in ("similar", "text") and f.value.strip()]
        need_embeddings = len(vector_filters) > 0

        select_cols = "id, path, type, date, date_ts, camera, lat, lon"
        if need_embeddings:
            select_cols += ", embedding"

        sql = f"SELECT {select_cols} FROM media WHERE " + " AND ".join(where_clauses)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return []

        candidates = []
        for r in rows:
            lat = float(r[6]) if r[6] is not None else None
            lon = float(r[7]) if r[7] is not None else None
            
            item = {
                "id": r[0],
                "path": r[1],
                "filename": os.path.basename(r[1]),
                "type": r[2],
                "date": r[3] or "Unknown",
                "date_ts": float(r[4]) if r[4] is not None else 0.0,
                "camera": r[5] or "Unknown",
                "lat": lat,
                "lon": lon,
                "similarity": 0.0
            }
            if need_embeddings:
                item["embedding"] = r[8]
            candidates.append(item)

        # Precise spherical distance check
        if location_filters:
            filtered_candidates = []
            for c in candidates:
                if c["lat"] is None or c["lon"] is None:
                    continue
                match_all = True
                for t_lat, t_lon, rad in location_filters:
                    if haversine_miles(t_lat, t_lon, c["lat"], c["lon"]) > rad:
                        match_all = False
                        break
                if match_all:
                    filtered_candidates.append(c)
            candidates = filtered_candidates

        # Semantic and Visual Vector Similarity Filtering
        if vector_filters:
            for f in vector_filters:
                q_vec = None
                thresh = 0.14 if f.type == "text" else 0.50
                if f.threshold is not None and f.threshold > 0:
                    thresh = float(f.threshold)

                if f.type == "text":
                    emb = get_text_embedding(f.value.strip())
                    if emb:
                        q_vec = np.array(emb, dtype=np.float32).flatten()
                elif f.type == "similar":
                    try:
                        target_id = int(f.value)
                        async with db.execute("SELECT embedding FROM media WHERE id = ?", (target_id,)) as cursor:
                            emb_row = await cursor.fetchone()
                        if emb_row and emb_row[0]:
                            q_vec = np.array(json.loads(emb_row[0]), dtype=np.float32).flatten()
                    except Exception:
                        pass

                if q_vec is None:
                    continue

                v_norm = np.linalg.norm(q_vec)
                if v_norm > 0:
                    q_vec = q_vec / v_norm

                surviving = []
                for item in candidates:
                    emb_str = item.get("embedding")
                    if not emb_str:
                        continue
                    try:
                        vec = np.array(json.loads(emb_str), dtype=np.float32).flatten()
                        u_norm = np.linalg.norm(vec)
                        if vec.shape[0] == q_vec.shape[0] and u_norm > 0:
                            sim = float(np.dot(vec, q_vec) / u_norm)
                            if sim >= thresh:
                                item["similarity"] = max(item["similarity"], sim)
                                surviving.append(item)
                    except Exception:
                        continue
                candidates = surviving

    # Sorting
    sort_mode = req.sort_by
    if sort_mode == "date_asc":
        candidates.sort(key=lambda x: x["date_ts"])
    elif sort_mode == "camera":
        candidates.sort(key=lambda x: (x["camera"].lower(), -x["date_ts"]))
    elif sort_mode == "relevance" and need_embeddings:
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
    else:  # Default: date_desc
        candidates.sort(key=lambda x: x["date_ts"], reverse=True)

    output = []
    for item in candidates[:req.limit]:
        item.pop("embedding", None)
        output.append(item)

    return output


@app.get("/media/{media_id}")
async def serve_media(media_id: int):
    """Serves media files, dynamically converting HEIC/unsupported formats to full JPEGs in-memory."""
    async with aiosqlite.connect(DB_PATH, timeout=30.0) as db:
        async with db.execute("SELECT path, type FROM media WHERE id = ?", (media_id,)) as cursor:
            row = await cursor.fetchone()
            if row and os.path.exists(row[0]):
                path, m_type = row[0], row[1]
                ext = os.path.splitext(path)[1].lower()
                if m_type in ("photo", "360_photo") and ext in ('.heic', '.heif', '.tiff', '.tif', '.bmp'):
                    full_bytes = generate_dynamic_full_image_bytes(path)
                    if full_bytes:
                        return Response(content=full_bytes, media_type="image/jpeg")
                
                # Stream .insv as video/mp4 and .insp as image/jpeg so browsers play/view them natively
                if ext == '.insv':
                    return FileResponse(path, media_type="video/mp4")
                if ext == '.insp':
                    return FileResponse(path, media_type="image/jpeg")

                return FileResponse(path)
            return Response(status_code=404)
