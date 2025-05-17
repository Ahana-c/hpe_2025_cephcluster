import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename

from utils import train_models, generate_analysis_plots, predict_recovery_time, predict_replication_rate

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__, template_folder='uploads/templates', static_folder='uploads/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

df = None  # Global dataset


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
            except Exception as e:
                flash(f'Error loading file: {e}')
                return redirect(request.url)

            try:
                train_models(df)
                plots = generate_analysis_plots(df)
            except Exception as e:
                flash(f'Error processing data: {e}')
                return redirect(request.url)

            return render_template('results.html', eda_plots=plots)
        else:
            flash('Allowed file types are csv, xlsx')
            return redirect(request.url)

    return render_template('index.html')


@app.route('/predict_recovery', methods=['POST'])
def predict_recovery():
    form_data = request.form.to_dict()
    try:
        prediction, accuracy = predict_recovery_time(form_data)
    except Exception as e:
        flash(f'Prediction error: {e}')
        return redirect(url_for('index'))

    return render_template('prediction.html', title='Recovery Time Prediction',
                           prediction=prediction, accuracy=accuracy)


@app.route('/predict_replication', methods=['POST'])
def predict_replication():
    form_data = request.form.to_dict()
    try:
        prediction, accuracy = predict_replication_rate(form_data)
    except Exception as e:
        flash(f'Prediction error: {e}')
        return redirect(url_for('index'))

    return render_template('prediction.html', title='Replication Rate Prediction',
                           prediction=prediction, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
