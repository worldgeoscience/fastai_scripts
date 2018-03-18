
# coding: utf-8

# ## Introduction to our first task: 'Dogs vs Cats'

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


PATH = "data/dogscats/"
sz=224
batch_size=64


# In[3]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input


# In[4]:


train_data_dir = f'{PATH}train'
validation_data_dir = f'{PATH}valid'


# In[5]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=False,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')


# In[6]:


base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)


# In[7]:


model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers: layer.trainable = False
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[8]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(train_generator, train_generator.n // batch_size, epochs=3, workers=4,\n        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)')


# In[9]:


split_at = 140
for layer in model.layers[:split_at]: layer.trainable = False
for layer in model.layers[split_at:]: layer.trainable = True
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])    


# In[10]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(train_generator, train_generator.n // batch_size, epochs=1, workers=3,\n        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)')

