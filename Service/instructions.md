# Final Folder Structure

Ensure your project folder looks like this:

```
project-folder/
│
├── main_12_asset.py
├── requirements.txt
├── Dockerfile
├── swagger-ui/
│   └── dist/ (Swagger UI files go here)
```

---

# Build and Run Docker Container

### Explanation of the Docker Compose File:

- **`services`**: Defines the containerized services (in this case, the FastAPI app).
- **`build`**: Points to the current directory (where the Dockerfile is located) to build the image.
- **`ports`**: Maps port `8000` on the container to port `8000` on the host machine.
- **`volumes`**: Mounts the `swagger-ui` folder from the host to the container to serve the offline Swagger UI files.
- **`restart`**: Ensures the container restarts automatically if it crashes.

---

### Steps to Run with Docker Compose:

#### Build and Run Containers:

**For the first time:**
```bash
docker-compose up --build -d
```

**For subsequent runs:**
```bash
docker-compose up -d
```

#### Stop Containers:

Use `Ctrl+C` or run:
```bash
docker-compose down
```

---

### Access the Application:

- **API**: [http://localhost:8000](http://localhost:8000)
- **Swagger UI (offline)**: [http://localhost:8000/docs](http://localhost:8000/docs)

