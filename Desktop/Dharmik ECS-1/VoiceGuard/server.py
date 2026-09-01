import uvicorn
from main import app

if __name__ == "__main__":
    print("\n=======================================================")
    print("  VoiceGuard Enterprise AI Deepfake Detection Engine")
    print("  Dashboard: http://127.0.0.1:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
