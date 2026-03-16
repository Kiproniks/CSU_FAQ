from app.web_app import app


if __name__ == "__main__":
    # Легковесный запуск локального веб-демо.
    print("Starting web demo on http://127.0.0.1:8000", flush=True)
    app.run(host="127.0.0.1", port=8000, debug=False)

