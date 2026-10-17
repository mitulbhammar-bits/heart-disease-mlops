"""Create a short, captioned walkthrough of real local artifacts (not deployment proof)."""
import json
from pathlib import Path
import subprocess
import textwrap

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 1280, 720, 12


def font(size):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def slide(title, lines, image_path=None):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#eef3f6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 85), fill="#16465e")
    draw.text((42, 22), title, fill="white", font=font(30))
    y = 112
    for line in lines:
        for wrapped in textwrap.wrap(line, width=86):
            draw.text((42, y), wrapped, fill="#253745", font=font(23))
            y += 32
        y += 9
    if image_path:
        source = Image.open(ROOT / image_path).convert("RGB")
        source.thumbnail((1180, max(50, 640-y)))
        image.paste(source, ((WIDTH-source.width)//2, y))
    draw.rectangle((0, 675, WIDTH, HEIGHT), fill="#dce9ef")
    draw.text((42, 685), "Real local artifacts | Docker, Kubernetes and hosted CI evidence remain pending",
              fill="#16465e", font=font(19))
    return image


def main():
    metadata = json.loads((ROOT / "models/model_metadata.json").read_text())
    api = json.loads((ROOT / "reports/evidence/api_verification.json").read_text())
    status = json.loads((ROOT / "reports/verification_status.json").read_text())
    scenes = [
        ("Heart Disease MLOps: local artifact walkthrough", [
            "Official UCI data -> EDA -> training -> tracked experiments -> reusable API.",
            "This video displays actual saved results. It is not a Docker/Kubernetes screen recording."],
         "reports/figures/architecture.png"),
        ("1. Acquire and clean the official data", [
            "303 patients; 13 predictors; binary target: diagnosis > 0.",
            "Four missing vessel values and two missing thal values; imputation stays inside CV."],
         "reports/figures/class_balance_and_missingness.png"),
        ("2. Compare models using training-only CV", [
            "Logistic Regression: C = 0.1, 1, 10. Random Forest: depth = 3, 6, unrestricted.",
            f"Winner: {metadata['selected_model']} {metadata['selected_parameters']}; CV ROC-AUC {metadata['cv_roc_auc']:.3f}."],
         "reports/figures/model_comparison.png"),
        ("3. Evaluate the untouched holdout", [
            "242 training rows; 61 holdout rows. Threshold 0.5; no holdout tuning.",
            "Actual metrics: " + "; ".join(f"{k} {v:.3f}" for k, v in metadata['holdout_metrics'].items()),
            "Small historical cohort: this demonstration is not clinical validation."],
         "reports/figures/holdout_confusion_matrix.png"),
        ("4. Inspect actual MLflow experiments", [
            "Six candidates plus one final holdout run; parameters, metrics, pipelines and plots.",
            "Screenshot captured from the running local MLflow interface."],
         "screenshots/mlflow_experiments.png"),
        ("5. Verify the live HTTP API", [
            "GET /health, /ready and /metrics passed; invalid JSON returned HTTP 422.",
            "Real POST /predict: " + json.dumps(api['predict']['body'])],
         "screenshots/local_api.png"),
        ("6. Verify behavior and prepare automated delivery", [
            "Actual local test result: " + status['tests'] + ". Lint passed.",
            "Data, serialization, missing values, API validation, readiness and metrics are tested.",
            "GitHub Actions is supplied for lint -> tests -> training -> Docker smoke test.",
            "A real hosted CI run still needs your repository."], None),
        ("7. Complete the remaining submission evidence", [
            "Run scripts/verify_docker.sh and scripts/deploy_minikube.sh with Docker/Minikube installed.",
            "Verify the LoadBalancer and show Grafana after sending real requests.",
            "Add your repo URL, capture real screenshots and record your narrated full pipeline.",
            "Rebuild the 10-page report and check reports/submission_checklist.md."], None),
    ]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output = ROOT / "reports/local_pipeline_walkthrough.mp4"
    process = subprocess.Popen([ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
                                "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
                                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(output)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for title, lines, picture in scenes:
        frame = slide(title, lines, picture).tobytes()
        for _ in range(FPS*10):
            process.stdin.write(frame)
    process.stdin.close()
    error = process.stderr.read().decode()
    if process.wait() != 0:
        raise RuntimeError(error)
    print(output)


if __name__ == "__main__":
    main()
