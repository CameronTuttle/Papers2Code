from keras.applications.imagenet_utils import preprocess_input, decode_predictions


class Show(object):

    @staticmethod
    def show_summary(model):
        print("[INFO] Model Summary: ")
        model.summary()
        return

    @staticmethod
    def show_config(model):
        print("[INFO] Model Config: ")
        config = model.get_config()
        for i, conf in enumerate(config):
            print(i, conf.__str__())
        return

    @staticmethod
    def show_evaluation(model, x_test, y_test, verbose=0):
        score = model.evaluate(x_test, y_test, verbose=verbose)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        return

    @staticmethod
    def show_predictions(model, img):
        preds = model.predict(img)
        print('{}'.format(model.metrics_names[1]), 'Prediction: ', decode_predictions(preds))
        return
