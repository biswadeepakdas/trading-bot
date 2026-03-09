#!/bin/bash
# Deploy Trading Bot to Google Cloud Run
#
# Prerequisites:
#   1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Run: gcloud auth login
#   3. Run: gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh

set -e

# ── Configuration ──
PROJECT_ID=$(gcloud config get-value project)
REGION="asia-south1"  # Mumbai (closest to Indian markets)
SERVICE_NAME="trading-bot"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "========================================"
echo "  Deploying Trading Bot to Cloud Run"
echo "  Project: ${PROJECT_ID}"
echo "  Region:  ${REGION}"
echo "========================================"

# Step 1: Build and push Docker image
echo ""
echo "[1/3] Building Docker image..."
gcloud builds submit --tag "${IMAGE_NAME}" --timeout=1200

# Step 2: Deploy to Cloud Run
echo ""
echo "[2/3] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    --memory 2Gi \
    --cpu 2 \
    --timeout 900 \
    --max-instances 1 \
    --allow-unauthenticated \
    --set-env-vars "FTP_HOST=ftpupload.net,FTP_USER=ezyro_41347592,FTP_PASS=524656b51"

# Get the service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format="value(status.url)")
echo ""
echo "Service deployed at: ${SERVICE_URL}"

# Step 3: Create Cloud Scheduler job (runs daily at 8:50 AM IST = 3:20 AM UTC)
echo ""
echo "[3/3] Setting up daily scheduler..."
gcloud scheduler jobs delete "${SERVICE_NAME}-daily" --location="${REGION}" --quiet 2>/dev/null || true
gcloud scheduler jobs create http "${SERVICE_NAME}-daily" \
    --location="${REGION}" \
    --schedule="50 8 * * 1-5" \
    --time-zone="Asia/Kolkata" \
    --uri="${SERVICE_URL}/" \
    --http-method=POST \
    --attempt-deadline=900s \
    --description="Run trading bot predictions daily at 8:50 AM IST (before market opens)"

echo ""
echo "========================================"
echo "  DEPLOYMENT COMPLETE!"
echo "========================================"
echo ""
echo "  Cloud Run URL:  ${SERVICE_URL}"
echo "  Dashboard:      https://tradingbot.unaux.com"
echo "  Schedule:       Mon-Fri at 8:50 AM IST"
echo ""
echo "  To trigger manually:"
echo "    curl -X POST ${SERVICE_URL}/"
echo ""
echo "  To view logs:"
echo "    gcloud run logs read --service=${SERVICE_NAME} --region=${REGION}"
echo ""
