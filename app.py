from flask import Flask, request, render_template, redirect, url_for
import os
from extract import clear_database, load_documents, split_documents, add_to_chroma
from query_data import query_rag

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "reset" in request.form:
            clear_database()
            return redirect(url_for("index"))
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file.save(os.path.join("uploads", file.filename))
            documents = load_documents()
            chunks = split_documents(documents)
            add_to_chroma(chunks)
            return redirect(url_for("index"))
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    query_text = request.form["query"]
    response = query_rag(query_text)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
