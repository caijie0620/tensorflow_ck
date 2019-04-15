import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
from datasets.dataloader import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


# Path to the textfiles for the trainings and validation set
root = '/home/jie/PycharmProjects/tensorflow_ck/data/cropped_256'
train_file = 'image_lists/set1/train.txt'
val_file = 'image_lists/set1/val.txt'
test_file = 'image_lists/set1/test.txt'

# Learning params
epochs = 30
learning_rate = 0.001
learning_rate_decay = 1e-4
batch_size = 60
num_classes = 7


def main():

    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=SGD(lr=learning_rate, decay=learning_rate_decay, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    for e in range(epochs):

        train_data = ImageDataGenerator(root, train_file, 'training', batch_size, num_classes, shuffle=False)

        val_data = ImageDataGenerator(root, val_file, 'inference', batch_size, num_classes)
        test_data = ImageDataGenerator(root, test_file, 'inference', batch_size, num_classes)

        print('Starting epoch %02d......' % (e,))

        total_loss = 0
        total_acc = 0
        for iteration, (xs, ys) in enumerate(tfe.Iterator(train_data.dataset)):
            output = model.train_on_batch(xs, ys)
            train1 = 'Train of iter %02d, ' % (iteration,)
            train2 = "Loss %.2f " % (output[0],)
            train3 = "Acc %.2f" % (output[1],)
            print(train1 + train2 + train3)
            total_loss += output[0]
            total_acc += output[1]
        print("\n")
        print("Train Ave_Loss: %.2f, Ave_Acc:  %.2f" % (total_loss / (iteration + 1), total_acc / (iteration + 1)))

        total_loss = 0
        total_acc = 0
        for iteration, (xs, ys) in enumerate(tfe.Iterator(val_data.dataset)):
            output = model.test_on_batch(xs, ys)
            total_loss += output[0]
            total_acc += output[1]
            '''
            val1 = 'Val of iter %02d, ' % (iter,)
            val2 = "Loss %.2f " % (output[0],)
            val3 = "Acc %.2f" % (output[1],)
            print(val1 + val2 + val3)
            '''
        print("Val   Ave_Loss: %.2f, Ave_Acc:  %.2f" % (total_loss / (iteration + 1), total_acc / (iteration + 1)))

        total_loss = 0
        total_acc = 0
        for iteration, (xs, ys) in enumerate(tfe.Iterator(test_data.dataset)):
            output = model.test_on_batch(xs, ys)
            total_loss += output[0]
            total_acc += output[1]
            '''
            test1 = 'Test of iter %02d, ' % (iter,)
            test2 = "Loss %.2f " % (output[0],)
            test3 = "Acc %.2f" % (output[1],)
            print(test1 + test2 + test3)
            '''
        print("Test  Ave_Loss: %.2f, Ave_Acc:  %.2f \n" % (total_loss / (iteration + 1), total_acc / (iteration + 1)))


if __name__ == "__main__":
    main()
