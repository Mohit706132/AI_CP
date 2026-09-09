#!/bin/bash

echo "========================================================"
echo "  Starting Integrated Plant Suite App"
echo "========================================================"
echo ""

if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
elif command -v ipconfig &> /dev/null; then
    LOCAL_IP=$(ipconfig getifaddr en0)
else
    LOCAL_IP="127.0.0.1"
fi

echo "[INFO] Local Server URL:  http://localhost:8501"
echo "[INFO] Mobile Access URL: http://${LOCAL_IP}:8501"
echo ""
echo "[NOTE] Make sure your mobile phone is connected to the same Wi-Fi network!"
echo ""
echo "Launching Streamlit server..."
echo "========================================================"

streamlit run app.py --server.address=0.0.0.0 --server.port=8501
