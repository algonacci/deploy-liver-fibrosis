import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import helpers

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["MODEL_FILE"] = "keras_model.h5"
app.config["LABELS_FILE"] = "labels.txt"
app.secret_key = "supersecretkey"  # Needed for flashing messages

# Load model and labels once at startup
model, labels = helpers.load_resources(
    app.config["MODEL_FILE"], app.config["LABELS_FILE"]
)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        file = request.files["image"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run prediction
            class_name, confidence_score = helpers.predict(filepath, model, labels)

            # Render results to the template
            return render_template(
                "pages/index.html",
                result=class_name,
                confidence_score=confidence_score,
                image_path=filepath,
            )

        else:
            flash("Allowed file types are png, jpg, jpeg")
            return redirect(request.url)

    # For GET requests, simply render the index page
    return render_template("pages/index.html")


# Route to serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="uploads/" + filename))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
