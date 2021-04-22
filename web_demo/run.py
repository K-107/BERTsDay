# -*- coding: utf-8 -*-

from app.main import app
import webbrowser

if __name__ == "__main__":
    app.run("localhost",port=6006, debug=True)
    webbrowser.open("http://localhost:6006")