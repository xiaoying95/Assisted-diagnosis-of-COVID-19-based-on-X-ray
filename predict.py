import os
import json
import glob
import numpy as np

from model import shufflenet_v2_x1_0
import tensorflow as tf

def predict_web(img_path):
    im_height = 224
    im_width = 224
    num_classes = 3

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)  # 强制编码成3通道了
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, im_height, im_width)  # 按照规定的尺寸填补或者剪裁
    # 如果原图像尺寸大于目标图像尺寸，则在中心位置剪裁，反之则用黑色像素填充。
    img = (img - mean) / std

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = shufflenet_v2_x1_0(num_classes=num_classes)

    weights_path = './save_weights/shufflenetv2.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    diagnosis = class_indict[str(predict_class)]
    print(diagnosis)
    return diagnosis

if __name__ == '__main__':
    #main()
    pass
