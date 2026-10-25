"""Launch only on the local computer, with the current Python environment."""
from pathlib import Path
import subprocess,sys
try: import streamlit
except ImportError: raise SystemExit('Install requirements.txt in a Python 3.11+ virtual environment first.')
raise SystemExit(subprocess.call([sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('app.py')),'--server.address','127.0.0.1','--server.port','8773','--browser.gatherUsageStats','false']))
