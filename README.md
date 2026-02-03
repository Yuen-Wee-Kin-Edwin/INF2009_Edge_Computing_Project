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
```

## Edge Pi
```
cd edge_pi/scripts
python3 capture_publish.py
```
