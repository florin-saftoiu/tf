import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

(train_num_samples, train_w, train_h) = x_train.shape
(test_num_samples, test_w, test_h) = x_test.shape
print(f'Training on {train_num_samples} samples')
print(f'Testing on {test_num_samples} samples')

# normalize inputs
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(probability_model(x_test[:5]))