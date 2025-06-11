from root folder
. .\.venv\Scripts\Activate.ps1                                

front end:
 cd .\api\visual-search-app\
 npm start


back end:
cd api
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

