# INF2009 Edge Computing Project

## Setup
1. Create virtual environment.
```zsh
python3 -m venv .venv
```
2. Activate virtual environment.
```zsh
source .venv/bin/activate
```
3. Install dependencies.
```zsh
sudo ./scripts/install_dependencies.sh
```
```zsh
pip install -r requirements.txt
```
4. Run the app
```zsh
sudo ./scripts/run_app.sh
```
5. OPTIONAL: Run sqlite-web
```zsh
sqlite_web src/lab_monitor.db --host 0.0.0.0 --port 8080
```

## MQTT Hub Setup.
```zsh
sudo ./scripts/setup_mqtt_hub.sh
username -> edwin
password -> password
```

## Edge Pi
```
sudo apt update
sudo apt install python3-pip
sudo apt install python3-opencv

# Download the quantized MobileNet V2 SSD model
wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite

# Download the corresponding COCO label map
wget https://github.com/google-coral/test_data/raw/master/coco_labels.txt


# Install pip packages
mkdir -p ~/tmp
export TMPDIR=~/tmp
pip install --no-cache-dir -r requirements-edge.txt

cd edge_pi/scripts
python3 capture_publish.py
```
