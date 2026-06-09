import shutil
from pathlib import Path
import uvicorn
import webbrowser
import threading
import time
import os
import sys
import json
import multiprocessing
from main import app  # IMPORTANT: Import the app directly instead of using a string
import warnings

warnings.filterwarnings('ignore')

_HOST = '127.0.0.1'
_PORT = 5459
_LINK = f'https://{_HOST}:{_PORT}'



def open_browser():
    time.sleep(1.5)  # Give the server a moment to start
    webbrowser.open(_LINK)


if __name__ == "__main__":
    try:

        # Required for Windows executables using multithreading/multiprocessing
        multiprocessing.freeze_support()

        # Start the browser thread
        threading.Thread(target=open_browser, daemon=True).start()

        # Run the server passing the APP OBJECT, not the string "main:app"
        uvicorn.run(app, host=_HOST, port=_PORT)

    except Exception as run_e:
        print('run_e', run_e)
    finally:
        print('finally some rest')
        # try:
        #     shutil.rmtree(Path('tmp'), ignore_errors=True)

        # except Exception as final_error:
        #     print('final_error', final_error)