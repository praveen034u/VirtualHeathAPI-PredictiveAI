#!/bin/bash

PROJECT_ID="dark-bit-459802-t7"
REGION="us-central1"
REPO="predictive-fastapi-repo"
IMAGE_NAME="predictive-fastapi"
SERVICE_NAME="predictive-api"

# Authenticate with Google Cloud using the service account key
echo "$GCP_SA_KEY" | base64 -d > /tmp/gcp_sa_key.json
gcloud auth activate-service-account --key-file=/tmp/gcp_sa_key.json --project=$PROJECT_ID

gcloud config set project $PROJECT_ID
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME

gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \

# Clean up the key file
rm /tmp/gcp_sa_key.json
