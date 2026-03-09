# WEB_CAMERA_AUDIT

## Scope
This audit covers the web UI and API under `userVisualization/` that support local
video upload and tri-modal inference.

## Frontend
- Stack: static HTML + CSS + vanilla JS (no React/Vue/Next). Chart.js is loaded
  from a CDN for the gauge visualization.
- Route entry: single page served at `GET /` (no client-side router). Static
  assets are loaded from `/static` (`app.js`, `styles.css`).
- Upload / inference request:
  - `app.js` builds a `FormData` payload with field name `video` and sends
    `POST /api/analyze` via `fetch`.
  - Response is expected as JSON with a `result` object containing `final`,
    `audio`, `video`, `sync` plus `elapsed`. Errors read `detail`.
  - The UI is updated directly in JS after the response resolves.

## Backend
- Stack: FastAPI + Uvicorn, `python-multipart` for file uploads, Torch for model
  inference. CORS is wide-open to simplify local UI calls.
- API list:
  - `GET /` -> serves the static HTML page.
  - `GET /api/health` -> returns `{ "status": "ok" }`.
  - `POST /api/analyze` -> accepts a multipart `video` file, writes a temp file,
    runs tri-modal inference, and returns `{"result": ...}`.
- File upload: already implemented via `UploadFile` in `/api/analyze`.

## Shortest Browser -> Backend -> Result Path
Yes, it already exists.
1. User selects a local video (or drag-and-drop).
2. Frontend posts the file to `/api/analyze` as `multipart/form-data` with
   field name `video`.
3. Backend stores a temp file, preprocesses audio/video, runs the tri-modal
   model, and returns JSON.
4. Frontend renders the returned scores and labels.

## Planned Changes to Ship (for webcam capture support)
Frontend
- Add a webcam capture panel (record/start/stop, live preview) using
  `getUserMedia` + `MediaRecorder`.
- Convert the recorded `Blob` to a `File` and reuse the existing
  `POST /api/analyze` upload path so the UI works for both file and webcam.
- Add clear UX states for permission errors, recording duration limits, and
  upload progress.

Backend
- Keep `/api/analyze` as the primary upload endpoint (no new endpoint required).
- Add basic validation: MIME/type allowlist (`video/webm`, `video/mp4`, etc.),
  size limit, and a friendly 4xx response for unsupported formats.

Middleware / plumbing
- Add an upload size limit middleware and consistent error shaping so the UI can
  display predictable error messages for oversized or invalid uploads.

Notes
- The current pipeline already retries decoding for browser-recorded WebM clips
  by falling back to OpenCV; that is compatible with webcam recordings.
