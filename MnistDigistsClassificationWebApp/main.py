import os
from flask import Flask, request
import model.access.LoadPredictCls

app = Flask(__name__)

UPLOAD_FOLDER = './upload'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file_predict():
    '''Webservice that uploads an image for a digit and predicts its value.'''
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        loader = model.access.LoadPredictCls.LoadPredictCls()
        loader.set_path_model(r"../model")
        # image_file_path = r"C:\Users\admin\Documents\computer_vision_proj\sample.png"
        im = loader.read_image_from_path(path)
        im = loader.preprocess_data(im)
        p = loader.load_model_and_predict(im)
        print(p)
        from flask import jsonify
        data = {"prediction": str(p)}
        return jsonify(data)

    return '''
    <h1>Upload image to classify</h1>
    <p>Click submit to predict the uploaded handwritten digit</p>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''


@app.route('/demo/', methods=['GET', 'POST'])
def welcome():
    '''Welcom Webservice for the demo.'''
    return "Demo Classifying handwritten digits."


if __name__ == '__main__':
    '''Main to run webservice.'''
    app.run(host='0.0.0.0', port=105)
