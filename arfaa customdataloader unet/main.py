from model import *
from data import *
from customdatagenerator import *


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# these augmentation params not needed if "augment" function is hard-coded to do random transformations 
# data_gen_args = dict(rotation_range=0.2,
                    # width_shift_range=0.05,
                    # height_shift_range=0.05,
                    # shear_range=0.05,
                    # zoom_range=0.05,
                    # horizontal_flip=True,
                    # fill_mode='nearest')

train_path = '/Users/arfaa/Desktop/school/other/AI4G/project/Data/Training-Splitted/train/'
val_path = '/Users/arfaa/Desktop/school/other/AI4G/project/Data/Training-Splitted/val/'
batch_size = 8
dim = (640,640) # dimensions of images  **actual dim or 32,32,32 like customdatagen.py?
n_channels = 1

# using custom datagenerator:
train_gen = CustomDataGenerator(path = train_path, batch_size = batch_size, dim = dim, n_channels = n_channels, shuffle = True, augmentation = True)
val_gen = CustomDataGenerator(path = val_path, batch_size = batch_size, dim = dim, n_channels = n_channels, shuffle = True, augmentation = True)

model = unet(input_size=(*dim,1))
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit(train_gen, val_gen ,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
model.fit(x=train_gen,epochs=1,callbacks=[model_checkpoint])
#https://www.tensorflow.org/api_docs/python/tf/keras/Model


#no change?** not to val path?
testGene = testGenerator("data/membrane/test") # I don't think this needs to be changed (the function does not use ImageDataGenerator nor augmentations)
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)