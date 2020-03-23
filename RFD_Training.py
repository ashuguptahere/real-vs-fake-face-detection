# Including Essential Libraries
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.mobilenet import MobileNet preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input

# Required Parameters
train_path = 'real_and_fake_face/'                  # Train Path
epochs = 10                                         # Number of epochs
batch_size = 32                                     # Batch Size
n_classes = 1                                       # Number of categories
modelname = "RFD_{}.h5".format(int(time.time()))       # Saved Model Name

# Defining Mobilenet Architecture
# mobile = tensorflow.keras.applications.mobilenet.MobileNet()

# Defining Base Model
base_model = MobileNetV2(weights = 'imagenet',
                        include_top = False,
                        input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
# x = Dense(512, activation = 'relu')(x)
# x = Dropout(0.5)(x)
preds = Dense(n_classes, activation = 'sigmoid')(x)

model = Model(inputs = base_model.input, outputs = preds)

for i, layer in enumerate(model.layers):
    print(i, layer.name)

# Setting all layers as trainable
# for layer in model.layers:
#     layer.trainable = True

# Setting untrainable layers
for layer in model.layers[:60]:
    layer.trainable = False
for layer in model.layers[60:]:
    layer.trainable = True

# Train Test Generator
# train_datagenerator = ImageDataGenerator(rescale = 1/255.,
#                                          shear_range=0.2,
#                                          zoom_range=0.2,
#                                          horizontal_flip = True,
#                                          validation_split = 0.2)

train_datagenerator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                         validation_split = 0.1)

test_datagenerator = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagenerator.flow_from_directory(train_path,
                                                          target_size = (224, 224),
                                                          color_mode = 'rgb',
                                                          batch_size = batch_size,
                                                          class_mode = 'binary',
                                                          shuffle = True)

validation_generator = train_datagenerator.flow_from_directory(train_path,
                                                               target_size = (224, 224),
                                                               color_mode = 'rgb',
                                                               batch_size = batch_size,
                                                               class_mode = 'binary',
                                                               subset = 'validation')

print(train_generator.class_indices)
print(validation_generator.class_indices)

# Compiling the Model
model.compile(optimizer = "adam",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

# Training the Model
history = model.fit(train_generator,
                    validation_data = validation_generator,
                    epochs = epochs)

# Plotting the Graph
import pandas as pd
pd.DataFrame(history.history).plot()

# Saving the Model
model.save(modelname)