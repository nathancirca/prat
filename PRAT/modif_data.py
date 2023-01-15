import numpy as np
from utils import store_data_and_plot

# for scene in ['training/lely','training/limagne','training/marais1']:
#     new_img = np.load(scene+'_10.npy')
#     shape = new_img.shape
#     new_img = new_img.reshape((shape[0],shape[1],1))
#     for i in range(11,21):
#         add = np.load(scene+'_'+str(i)+'.npy')
#         add = add.reshape((shape[0],shape[1],1))
#         new_img = np.concatenate((new_img,add),2)
#     np.save(scene, new_img)

# new_img = np.load('test/groundtruth/marais2_20.npy')
# shape = new_img.shape
# new_img = new_img.reshape((shape[0],shape[1],1))
# for i in range(21,31):
#     add = np.load('test/groundtruth/marais2_'+str(i)+'.npy')
#     add = add.reshape((shape[0],shape[1],1))
#     new_img = np.concatenate((new_img,add),2)
# np.save('test/groundtruth/marais2', new_img)
        
im = np.load('data/training/lely.npy')
print(im.shape)
store_data_and_plot(im[:,:,0],255,"data/training/lely_im.npy")