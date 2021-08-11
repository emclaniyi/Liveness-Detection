import pickle
from preprocessing import extract_feature, load_feature

pickle_in = open('new_liveness.pkl', 'rb')
model = pickle.load(pickle_in)

data = 'new_fake.jpg'


def predict_spoof(file_path):
    extract_feature(file_path)
    new_img = load_feature('new_img_feature.npy')
    pred = model.predict([new_img])
    if pred == 0:
        print('Denied!!! T  his is a Fake Photo')
    else:
        print('Real Photo')
    return pred


print(predict_spoof(data))
