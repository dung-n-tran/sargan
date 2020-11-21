import matplotlib.pyplot as plt
from sargan_models import SARGAN
import os
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
import skimage.measure as ski_me
import time
import numpy as np
from cifar_helper import get_data, chunks
from sar_utilities import add_gaussian_noise
from alert_utilities import send_images_via_email

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

img_size = (224,224,1)
experiment_name = 'fashion_mnist_gaussian_corrupted'
output_dir = '/scratch/hle/data/outputs/'
trained_model_path = '/scratch/hle/data/trained_models/'
output_path = os.path.join(output_dir, experiment_name)
NUM_ITERATION = 85
BATCH_SIZE = 64
GPU_ID = 1
MAX_EPOCH = 50
LEARNING_RATE = 0.001
NOISE_STD_RANGE = [0.1, 0.3]
SAVE_EVERY_EPOCH = 5
plt.switch_backend('agg')

####
#GETTING IMAGES
####
train_filename = "data_batch_2"
test_filename = "test_batch"


def get_fashion_mnist_data(train_batch_size, val_batch_size):
    mnist = datasets.FashionMNIST(root='/scratch/hle/', train=True, transform=torchvision.transforms.ToTensor(), target_transform=None, download=False).data.float()
    
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])
    train_loader = DataLoader(datasets.FashionMNIST(root='/scratch/hle/', train=True, transform=data_transform, target_transform=None, download=False),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(datasets.FashionMNIST(root='/scratch/hle/', train=False, transform=data_transform, target_transform=None, download=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

def get_caltech_data():
    caltech_dir = '/scratch/hle/101_ObjectCategories'
    list_dirs = os.listdir(caltech_dir)
    list_dirs.sort()
    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'val':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    traindir = '/scratch/hle/caltech_101/train'
    validdir = '/scratch/hle/caltech_101/valid'
    testdir = '/scratch/hle/caltech_101/test'
    batch_size = BATCH_SIZE
    data = {
        'train':
        datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'val':
        datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
        'test':
        datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }
    return dataloaders



def main(args):
    model = SARGAN(img_size, BATCH_SIZE, img_channel=1)
    d_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.g_loss, var_list=model.g_vars)
    
    saver = tf.train.Saver(max_to_keep=20)
    
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    progress_bar = tqdm(range(MAX_EPOCH), unit="epoch")
    #list of loss values each item is the loss value of one ieteration
    train_d_loss_values = []
    train_g_loss_values = []
    
    
    #test_imgs, test_classes = get_data(test_filename)
    #imgs, classes = get_data(train_filename)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        #copies = imgs.astype('float32')
        #test_copies = test_imgs.astype('float32')
        for epoch in progress_bar:
            train_loader, val_loader = get_fashion_mnist_data(BATCH_SIZE, BATCH_SIZE)
            counter = 0
            epoch_start_time = time.time()
            #shuffle(copies)
            #divide the images into equal sized batches
            #image_batches = np.array(list(chunks(copies, BATCH_SIZE)))
            trainiter = iter(train_loader)
            for i in range (NUM_ITERATION):
                #getting a batch from the training data
                #one_batch_of_imgs = image_batches[i]
                features, labels = next(trainiter)
                features = features.data.numpy().transpose(0,2,3,1)
                #copy the batch
                features_copy = features.copy()
                #corrupt the images
                corrupted_batch = np.array([add_gaussian_noise(image, sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1])) for image in features_copy])
                
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:features, model.cond:corrupted_batch})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:features, model.cond:corrupted_batch})
                train_d_loss_values.append(m)
                train_g_loss_values.append(M)
                #print some notifications
                counter += 1
                if counter % 25 == 0:
                    print("\rEpoch [%d], Iteration [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, counter, time.time() - epoch_start_time, m, M), end="")
                
            # save the trained network
            if epoch % SAVE_EVERY_EPOCH == 0:
                save_path = saver.save(sess, trained_model_path + ('%s/'% (experiment_name)) +\
                                   "%s_model_%s.ckpt" % ( experiment_name, epoch+1))
                print("Model saved in file: %s" % save_path)
            
            
            ##### TESTING FOR CURRUNT EPOCH
            testiter = iter(val_loader)
            NUM_TEST_PER_EPOCH = 1
           
            #test_batches = np.array(list(chunks(test_copies, BATCH_SIZE)))
            #test_images = test_batches[0]
            sum_psnr = 0
            list_images = []
            for j in range(NUM_TEST_PER_EPOCH):
                features, labels = next(testiter)
                features = features.data.numpy()
                features = features.transpose(0,2,3,1)
                batch_copy = features.copy()
                #corrupt the images
                corrupted_batch = np.array([add_gaussian_noise(image, sd=np.random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1])) for image in batch_copy])


                gen_imgs = sess.run(model.gen_img, feed_dict={model.image:features, model.cond:corrupted_batch})
                print(features.shape, gen_imgs.shape)
                #if j %17 == 0: # only save 3 images 0, 17, 34
                list_images.append((features[0], corrupted_batch[0], gen_imgs[0]))
                list_images.append((features[17], corrupted_batch[17], gen_imgs[17]))                
                list_images.append((features[34], corrupted_batch[34], gen_imgs[34]))
                for i in range(len(gen_imgs)):
                    current_img = features[i]
                    recovered_img = gen_imgs[i]
                    sum_psnr += ski_me.compare_psnr(current_img, recovered_img, 1)
                #psnr_value = ski_mem.compare_psnr(test_img, gen_img, 1)
                #sum_psnr += psnr_value
            average_psnr = sum_psnr / 50
            
            epoch_running_time = time.time() - epoch_start_time
            ############### SEND EMAIL ##############
            rows = 1
            cols = 3
            display_mean = np.array([0.485, 0.456, 0.406])
            display_std = np.array([0.229, 0.224, 0.225])
            #image = std * image + mean
            imgs_1 = list_images[0]
            imgs_2 = list_images[1]
            imgs_3 = list_images[2]
            imgs_1 = display_std * imgs_1 + display_mean
            imgs_2 = display_std * imgs_2 + display_mean
            imgs_3 = display_std * imgs_3 + display_mean
            fig = plt.figure(figsize=(14, 4))
            ax = fig.add_subplot(rows, cols, 1)
            ax.imshow(imgs_1[0])
            ax.set_title("Original", color='grey')
            ax = fig.add_subplot(rows, cols, 2)
            ax.imshow(imgs_1[1])
            ax.set_title("Corrupted", color='grey')
            ax = fig.add_subplot(rows, cols, 3)
            ax.imshow(imgs_1[2])
            ax.set_title("Recovered", color='grey')
            plt.tight_layout()
            sample_test_file_1 = os.path.join(output_path, '%s_epoch_%s_batchsize_%s_1.jpg' % (experiment_name, epoch, BATCH_SIZE))
            if epoch % SAVE_EVERY_EPOCH == 0:
                plt.savefig(sample_test_file_1, dpi=300)
            
            fig = plt.figure(figsize=(14, 4))
            ax = fig.add_subplot(rows, cols, 1)
            ax.imshow(imgs_2[0])
            ax.set_title("Original", color='grey')
            ax = fig.add_subplot(rows, cols, 2)
            ax.imshow(imgs_2[1])
            ax.set_title("Corrupted", color='grey')
            ax = fig.add_subplot(rows, cols, 3)
            ax.imshow(imgs_2[2])
            ax.set_title("Recovered", color='grey')
            plt.tight_layout()
            sample_test_file_2 = os.path.join(output_path, '%s_epoch_%s_batchsize_%s_2.jpg' % (experiment_name, epoch, BATCH_SIZE))
            plt.savefig(sample_test_file_2, dpi=300)
 
            fig = plt.figure(figsize=(14, 4))
            ax = fig.add_subplot(rows, cols, 1)
            ax.imshow(imgs_3[0])
            ax.set_title("Original", color='grey')
            ax = fig.add_subplot(rows, cols, 2)
            ax.imshow(imgs_3[1])
            ax.set_title("Corrupted", color='grey')
            ax = fig.add_subplot(rows, cols, 3)
            ax.imshow(imgs_3[2])
            ax.set_title("Recovered", color='grey')
            plt.tight_layout()
            sample_test_file_3 = os.path.join(output_path, '%s_epoch_%s_batchsize_%s_3.jpg' % (experiment_name, epoch, BATCH_SIZE))
            plt.savefig(sample_test_file_3, dpi=300)
 

            attachments = [sample_test_file_1, sample_test_file_2, sample_test_file_3]

            email_alert_subject = "Epoch %s %s SARGAN" % (epoch+1, experiment_name.upper())
            
            email_alert_text = """
            Epoch [{}/{}]
            EXPERIMENT PARAMETERS:
                        Learning rate: {}
                        Batch size: {}
                        Max epoch number: {}
                        Training iterations each epoch: {}
                        noise standard deviation: {}
                        Running time: {:.2f} [s]
            AVERAGE PSNR VALUE ON 50 TEST IMAGES: {}
            """.format(epoch + 1, MAX_EPOCH, LEARNING_RATE, BATCH_SIZE, MAX_EPOCH, 
                      NUM_ITERATION, ((NOISE_STD_RANGE[0] + NOISE_STD_RANGE[1])/2), epoch_running_time, average_psnr)
            if epoch % SAVE_EVERY_EPOCH == 0: 
                send_images_via_email(email_alert_subject,
                 email_alert_text,
                 attachments,
                 sender_email="hieule246@gmail.com", 
                 recipient_emails=["hieule246@gmail.com"])
            plt.close("all")            
        
if __name__ == '__main__':
    main([])
