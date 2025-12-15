# API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

Check if the service is running and model is loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is down or model not loaded

---

### Predict Distress

Detect distress from multi-modal input.

**Endpoint:** `POST /predict`

**Request:**

Content-Type: `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `audio_file` | File | Audio file (WAV format, 16kHz recommended) |
| `hrv_data` | File | HRV data (JSON format with RR intervals) |
| `accel_data` | File | Accelerometer data (JSON format with x,y,z axes) |

**HRV Data Format (JSON):**
```json
{
  "rr_intervals": [800, 820, 795, 810, ...],
  "timestamp": "2024-12-16T10:30:00Z"
}
```

**Accelerometer Data Format (JSON):**
```json
{
  "x": [0.1, 0.2, 0.15, ...],
  "y": [0.05, 0.08, 0.06, ...],
  "z": [9.8, 9.79, 9.81, ...],
  "sampling_rate": 50,
  "timestamp": "2024-12-16T10:30:00Z"
}
```

**Response:**
```json
{
  "distress_probability": 0.85,
  "prediction": "distress",
  "confidence": 0.85,
  "inference_time_ms": 342.5,
  "explanation": {
    "audio_contribution": 0.6,
    "biometric_contribution": 0.4,
    "shap_values": {...}
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `distress_probability` | float | Probability of distress (0.0 - 1.0) |
| `prediction` | string | Binary prediction: "distress" or "no_distress" |
| `confidence` | float | Model confidence (0.0 - 1.0) |
| `inference_time_ms` | float | Time taken for inference in milliseconds |
| `explanation` | object | SHAP-based explanation (optional) |

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input format
- `413 Payload Too Large`: File size exceeds limit
- `422 Unprocessable Entity`: Invalid data format
- `500 Internal Server Error`: Server error

---

## Error Responses

**Format:**
```json
{
  "error": "Error message",
  "detail": "Detailed error description"
}
```

---

## Rate Limiting

- **Limit:** 100 requests per minute per IP
- **Response when exceeded:** `429 Too Many Requests`

---

## Example Usage

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "audio_file=@sample_audio.wav" \
  -F "hrv_data=@hrv_data.json" \
  -F "accel_data=@accel_data.json"
```

### Python (requests)

```python
import requests

url = "http://localhost:8000/predict"

files = {
    "audio_file": open("sample_audio.wav", "rb"),
    "hrv_data": open("hrv_data.json", "rb"),
    "accel_data": open("accel_data.json", "rb")
}

response = requests.post(url, files=files)
print(response.json())
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("audio_file", audioFile);
formData.append("hrv_data", hrvFile);
formData.append("accel_data", accelFile);

fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData
})
  .then(res => res.json())
  .then(data => console.log(data));
```

---

## Performance SLA

- **Target Response Time:** < 500ms
- **Availability:** 99.9% uptime
- **Max Request Size:** 10MB
