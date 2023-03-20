import unittest
import model.access.LoadPredictCls


class MyTestCase(unittest.TestCase):
    def test_access_and_prediction(self):
        '''unit test that reads a digit image and predicts its value. value should be equal to 3 for sample.png'''
        loader = model.access.LoadPredictCls.LoadPredictCls()
        loader.set_path_model(r"..\model")
        image_file_path = r"..\sample.png"
        im = loader.read_image_from_path(image_file_path)
        im = loader.preprocess_data(im)

        self.assertEqual(3, loader.load_model_and_predict(im))  


if __name__ == '__main__':
    help(MyTestCase)
    unittest.main()
