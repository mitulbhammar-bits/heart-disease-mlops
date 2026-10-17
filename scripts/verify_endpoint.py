"""Save actual HTTP results; works against Python, Docker or Kubernetes."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--output", default="reports/evidence/api_verification.json")
args = parser.parse_args()
base = args.url.rstrip("/")
results = {}
for endpoint in ("health", "ready", "metrics"):
    with urllib.request.urlopen(f"{base}/{endpoint}", timeout=30) as response:
        body = response.read().decode()
        results[endpoint] = {"status": response.status, "body": body}
        if endpoint == "metrics":
            assert "heart_api_requests_total" in body
        else:
            assert json.loads(body)["status"] in ("ok", "ready")
patient = Path("examples/patient.json").read_bytes()
request = urllib.request.Request(f"{base}/predict", data=patient,
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=30) as response:
    prediction = json.loads(response.read())
    assert prediction["prediction"] in (0, 1)
    assert 0 <= prediction["confidence"] <= 1
    results["predict"] = {"status": response.status, "body": prediction}
request = urllib.request.Request(f"{base}/predict", data=b'{"age": -1}',
                                 headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(request, timeout=30)
    raise AssertionError("Invalid request should return HTTP 422")
except urllib.error.HTTPError as error:
    assert error.code == 422
    results["invalid_request"] = {"status": error.code}
results["base_url"] = base
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2))
print(json.dumps({"base_url": base, "prediction": prediction, "invalid_request": 422}, indent=2))
