from keras.datasets import mnist

if __name__=='__main__':
    (train_images,train_labels), (test_images,test_label)=mnist.load_data()