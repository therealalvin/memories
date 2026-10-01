# Memories 📸🎥

Welcome to **Memories**, a self-hosted application designed to help you store, organize, and relive your favorite photos and videos in one private, secure place.

---

## What It Does
* **Centralized Media Hub:** Aggregates your personal photo and video libraries into a single, clean web interface.
* **Dockerized Deployment:** Runs entirely in containers for easy setup, isolation, and portability.
* **Self-Hosted Privacy:** Keeps your personal memories on your own hardware rather than relying on third-party cloud storage.

---

## Prerequisites
Before you begin, make sure you have the following installed on your machine:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

---

## Configuration (`docker-compose.yaml`)

To point the container to your local media, you need to map your host machine's directories (where your photos and videos live) into the container using volume mounts.

Create a `docker-compose.yaml` file in your project directory and configure it as follows:

```yaml
version: '3.8'

services:
  memories:
    image: therealalvin/memories:latest
    container_name: memories-app
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=UTC
    volumes:
      - /path/to/your/app/data:/app/data
      - /path/to/your/photos:/photos
      - /path/to/your/videos:/videos
```

### Path Customization Guide
Replace the placeholder paths on the left side of the colon (`:`) with your actual local directory paths:
* `/path/to/your/app/data`: Where the application stores its database and configuration files.
* `/path/to/your/photos`: The absolute path on your host machine where your **photos** are stored (e.g., `/home/username/Pictures` on Linux or `/Users/username/Pictures` on macOS).
* `/path/to/your/videos`: The absolute path on your host machine where your **videos** are stored (e.g., `/home/username/Videos`).

---

## How to Run It

1. Open your terminal in the directory containing your `docker-compose.yaml` file.
2. Start the application in detached mode by running:
   ```bash
   docker compose up -d
   ```
3. Once the containers are up and running, open your web browser and navigate to:
   ```text
   http://localhost:3000
   ```
4. To view logs and troubleshoot any startup issues, run:
   ```bash
   docker compose logs -f
   ```
5. To stop the application when needed, run:
   ```bash
   docker compose down