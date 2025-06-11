import os
import threading
import subprocess

def run_command(name, command, cwd, env):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in process.stdout:
        print(f"[{name}] {line}", end='')

# Base project directory
base_path = r"C:\Users\fatim\OneDrive - De Montfort University\Final-Msc-Project-Fatima"
venv_path = os.path.join(base_path, ".venv")
venv_scripts = os.path.join(venv_path, "Scripts")

# Clone current env and simulate virtualenv activation
env_vars = os.environ.copy()
env_vars["VIRTUAL_ENV"] = venv_path
env_vars["PATH"] = f"{venv_scripts};{env_vars['PATH']}"

# Backend thread
backend_thread = threading.Thread(
    target=run_command,
    args=(
        "Backend",
        "uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000",
        os.path.join(base_path, "api"),
        env_vars
    )
)

# Frontend thread
frontend_thread = threading.Thread(
    target=run_command,
    args=(
        "Frontend",
        "npm start",
        os.path.join(base_path, "api", "visual-search-app"),
        env_vars
    )
)

# Start both
backend_thread.start()
frontend_thread.start()

# Wait for both to finish
backend_thread.join()
frontend_thread.join()

'''
start venv


. .\.venv\Scripts\Activate.ps1
python run_env.py


backend

cd api
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000


frontend

cd api\visual-search-app
npm start

'''

