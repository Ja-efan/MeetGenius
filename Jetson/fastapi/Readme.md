# Fast API App 기본구조

```shell

# fastapi/
uvicorn app.main:app --port 8000 --reload
```

```

    📦fastapi
    ┣ 📂app
    ┃ ┣ 📂api
    ┃ ┃ ┗ 📂routes
    ┃ ┣ 📂core
    ┃ ┣ 📂schemes
    ┃ ┣ 📂services
    ┃ ┣ 📂utils
    ┃ ┣ 📜dependencies.py
    ┃ ┗ 📜main.py
    ┣ 📂dist
    ┣ 📂vector_db
    ┣ 📂venv
    ┣ 📜Dockerfile
    ┣ 📜Readme.md
    ┣ 📜requirements-jetson.txt
    ┣ 📜requirements-macos.txt
    ┗ 📜requirements-windows.txt
```