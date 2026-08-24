# MiniLM Vector Search UI

From the project root, build the C++ MiniLM executable and start Flask:

```powershell
cmake --build build --config Release --target test_onnx
python -m pip install -r web/requirements.txt
python web/app.py
```

Open `http://127.0.0.1:5000` in a browser. Add custom texts, enter a query,
and select **Find similar texts**. Documents are stored only in memory, so they
reset when the Flask process stops.

The web app uses `build/Release/test_onnx.exe --embed <text>` to generate its
MiniLM embeddings. To use an executable at a different location, set the
`VECTOR_DB_EXECUTABLE` environment variable to its full path before starting
Flask.
