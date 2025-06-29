import numpy as np
from extract_features_deep import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load Data
X, y = load_data("Audio_Speech_Actors_01-24")
X = X[..., np.newaxis]  # Add channel for CNN

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 130, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))  # 8 emotions

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

# Save Model
model.save("cnn_emotion_model.h5")
