from keras.models import load_model, model_from_json


class Backup(object):

    @staticmethod
    def save_all(model, url='./url/2/model_all_file.h5'):
        model.save(filepath=url, overwrite=True)
        return

    @staticmethod
    def load_model_all(url='./url/2/model_all_file.h5'):
        model = load_model(url)
        return model

    @staticmethod
    def save_architecture(model, url='./url/2/model_arch_file.json'):
        # serialize model to JSON
        model_json = model.to_json()
        with open(url, 'w') as json_file:
            json_file.write(model_json)
        return

    @staticmethod
    def load_model_architecture(url='./url/2/model_arch_file.json'):
        # load json and create model
        json_file = open(url, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        return model

    @staticmethod
    def save_model_weights(model, url='./url/2/model_weights.h5'):
        # serialize weights to HDF5
        model.save_weights(url)
        return

    @staticmethod
    def load_model_weights(model, url='./url/2/model_all_weights.h5'):
        model.load_weights(url)
        return model

    @staticmethod
    def read_h5_file(url='./url/2/file.h5', dicc='X_data'):
        import h5py
        with h5py.File(url, 'r') as f:
            data = f[dicc]
        return data
