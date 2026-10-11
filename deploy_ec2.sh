#!/usr/bin/env bash
set -e

echo "=== 1. Updating packages and installing Python ==="
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

echo "=== 2. Setting up Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== 3. Installing lightweight dashboard dependencies ==="
pip install --upgrade pip
pip install -r requirements-app.txt

echo "=== 4. Starting Unified Streamlit Portal in background ==="
# Kill any existing streamlit processes
pkill -f streamlit || true

# Start unified portal on port 8501
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1 &

IP=$(curl -s http://checkip.amazonaws.com || echo "YOUR-EC2-PUBLIC-IP")

echo "=========================================================="
echo " National Waste Compliance Audit Portal is running!"
echo " Access the live portal at: http://${IP}:8501"
echo " (Includes both Phantom Plastic and Biomedical Waste Gap)"
echo " Check logs with: tail -f streamlit.log"
echo "=========================================================="
