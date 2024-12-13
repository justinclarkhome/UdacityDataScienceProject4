from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os, sys
import plotly.express as px
import plotly.io as pio

sys.path.append('../..')

from justin_model import build_model


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image (example: get image size)
        image = Image.open(file_path)
        image_info = f"Image size: {image.size[0]}x{image.size[1]}"
        
        # Create a simple Plotly figure as an example
        fig = px.imshow(image)
        plot_html = pio.to_html(fig, full_html=False)
        
        return render_template('index.html', filename=filename, image_info=image_info, plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
