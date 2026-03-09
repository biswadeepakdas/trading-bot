#!/usr/bin/env python3
"""
Google Cloud Run entry point for Trading Bot.

Exposes an HTTP endpoint that:
1. Runs ML predictions (run_prediction.py)
2. Uploads generated HTML to ProFreeHost via FTP
3. Returns status JSON

Triggered by Cloud Scheduler on a daily cron schedule.
"""

import os
import sys
import json
import ftplib
import traceback
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# FTP config for ProFreeHost
FTP_HOST = os.environ.get("FTP_HOST", "ftpupload.net")
FTP_USER = os.environ.get("FTP_USER", "ezyro_41347592")
FTP_PASS = os.environ.get("FTP_PASS", "524656b51")


def upload_to_profreehost(html_path):
    """Upload generated index.html to ProFreeHost via FTP."""
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login(FTP_USER, FTP_PASS)
    ftp.cwd("/htdocs")
    file_size = os.path.getsize(html_path)
    with open(html_path, "rb") as f:
        ftp.storbinary("STOR index.html", f)
    ftp.quit()
    return file_size


@app.route("/", methods=["GET", "POST"])
def run():
    """Main endpoint — runs predictions and uploads dashboard."""
    start = datetime.now()
    result = {
        "status": "started",
        "timestamp": start.isoformat(),
        "steps": []
    }

    try:
        # Step 1: Run predictions
        result["steps"].append("Running ML predictions...")

        # Import and run the prediction pipeline
        from run_prediction import run_predictions
        run_predictions()
        result["steps"].append("Predictions complete")

        # Step 2: Find the generated HTML
        html_path = os.path.join("public", "index.html")
        if not os.path.exists(html_path):
            result["status"] = "error"
            result["error"] = "public/index.html not generated"
            return jsonify(result), 500

        html_size = os.path.getsize(html_path)
        result["steps"].append(f"HTML generated: {html_size:,} bytes")

        # Step 3: Upload to ProFreeHost
        result["steps"].append("Uploading to ProFreeHost...")
        uploaded_size = upload_to_profreehost(html_path)
        result["steps"].append(f"Uploaded {uploaded_size:,} bytes to tradingbot.unaux.com")

        # Done
        elapsed = (datetime.now() - start).total_seconds()
        result["status"] = "success"
        result["elapsed_seconds"] = round(elapsed, 1)
        result["dashboard_url"] = "https://tradingbot.unaux.com"
        return jsonify(result), 200

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["elapsed_seconds"] = round(elapsed, 1)
        return jsonify(result), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
