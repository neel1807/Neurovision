from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from main import main

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

SEVERITY_INFO = {
    0: {"name": "No DR","description": "Your retina appears healthy.","steps": ["Routine yearly eye check.","Maintain blood sugar levels."]},
    1: {"name": "Mild DR","description": "Early blood vessel changes detected.","steps": ["Check eyes every 6-12 months.","Improve sugar & BP control."]},
    2: {"name": "Moderate DR","description": "Noticeable retinal changes.","steps": ["Monitor every 3-6 months.","Watch for vision distortion."],"specialist_search": "retina specialist"},
    3: {"name": "Severe DR","description": "High risk of vision loss.","steps": ["Urgent retina specialist visit required."],"specialist_search": "retina specialist"},
    4: {"name": "Proliferative DR","description": "Advanced stage. Immediate care needed.","steps": ["Emergency ophthalmologist treatment required."],"specialist_search": "ophthalmologist retina surgeon"}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    prediction = None
    filename = None
    severity_info = None
    
    if request.method == "POST":
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("Please upload an image.", "danger")
            return render_template("dashboard.html")

        file = request.files['file']
        if not allowed_file(file.filename):
            flash("Invalid file type. Upload an image.", "danger")
            return render_template("dashboard.html")

        filename = secure_filename(file.filename)
        import time
        stamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{stamp}{ext}"

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            value, classes = main(filepath)
            prediction = f"{value} - {classes}"
            severity_info = SEVERITY_INFO.get(value, None)
            flash("Analysis complete.", "success")
        except:
            flash("Error analyzing image. Try another image.", "danger")
            os.remove(filepath)
            filename = None

    return render_template("dashboard.html", prediction=prediction, filename=filename, severity_info=severity_info)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
