import json
import h5py
import tensorflow as tf
import numpy as np

class Model40Dataset:
    def __init__(self, data_path, split='train', batch_size=32, normalize=True, data_augmentation=True):
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.label_map = self.get_labels()
        self.dataset = self.load_data()

    ################################ File Loading ################################

    def get_filenames(self):
        with open(f'{self.data_path}{self.split}_files.txt', 'r') as file:
            files = file.read().splitlines()
            files = [self.data_path + line.split('/')[-1] for line in files]
        return files

    def get_labels(self):
        with open(f'{self.data_path}shape_names.txt', 'r') as file:
            classes = file.read().splitlines()
            classes = {class_: i for i, class_ in enumerate(classes)}
        return classes

    def dataset_from_h5(self, h5_file, json_file):
        with h5py.File(h5_file, 'r') as h5f, open(json_file, 'r') as jf:
            data = h5f['data'][()] 
            object_names = json.load(jf)
            classes = [obj.split('/')[0] for obj in object_names] 
            labels = [self.label_map[class_] for class_ in classes]

            data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
            labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices((data_tensor, labels_tensor))

            return dataset

    def load_data(self):
        filenames = self.get_filenames()
        combined_dataset = None


        for h5_file in filenames:
            name, _ = h5_file.split('.')
            json_file = name[:-1] + "_" + name[-1] + "_" "id2file.json"

            dataset = self.dataset_from_h5(h5_file, json_file)
            if combined_dataset is None:
                combined_dataset = dataset
            else:
                combined_dataset = combined_dataset.concatenate(dataset)
        return combined_dataset

    ################################ Preprocessing ################################

    def preprocess(self, point_cloud, label):
        if self.normalize:
            point_cloud, label = self.add_normalization(point_cloud, label)
        if self.data_augmentation:
            point_cloud, label = self.add_rotation(point_cloud, label)
            point_cloud, label = self.add_jitter(point_cloud, label)
        return point_cloud, label

    def add_normalization(self, point_cloud, label):
        centroid = tf.reduce_mean(point_cloud, axis=0)
        point_cloud_centered = point_cloud - centroid
        max_dist = tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(point_cloud_centered), axis=1)))
        point_cloud_normalized = point_cloud_centered / max_dist
        return point_cloud_normalized, label

    def add_jitter(self, point_cloud, label):
        jitter = 0.001
        noise = tf.random.uniform(tf.shape(point_cloud), minval=-jitter, maxval=jitter)
        point_cloud += noise
        return point_cloud, label

    def add_rotation(self, point_cloud, label):
        theta = tf.random.uniform((), 0, 2 * tf.constant(np.pi))
        rotation_matrix = tf.reshape(tf.stack([tf.cos(theta),   0.0,    tf.sin(theta),
                                            0.0          ,   1.0,              0.0,
                                            -tf.sin(theta),  0.0,  tf.cos(theta)]), (3, 3))
        rotated_point_cloud = tf.linalg.matmul(point_cloud, rotation_matrix)
        return rotated_point_cloud, label
    
     ################################ Dataset Creation ################################

    def create_dataset(self):
        self.dataset = self.dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) 
        return self.dataset



class ShapeNetDataset:
    def __init__(self, data_path, split='train', batch_size=32, normalize=True, data_augmentation=True):
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.label_map = self.get_labels()
        self.dataset = self.load_data()

    ################################ File Loading ################################

    # this method returns a list of file names
    def get_filenames(self):
        with open(f'{self.data_path}{self.split}_files.txt', 'r') as file:
            files = file.read().splitlines()
            files = [self.data_path + line.split('/')[-1] for line in files]
        return files

    def get_labels(self):
        with open(f'{self.data_path}shape_names.txt', 'r') as file:
            classes = file.read().splitlines()
            classes = {class_: i for i, class_ in enumerate(classes)}
        return classes

    def dataset_from_h5(self, h5_file, json_file):
        with h5py.File(h5_file, 'r') as h5f, open(json_file, 'r') as jf:
            data = h5f['data'][()]      # For modelnet40, each pointcloud is 2048 points. For shapenet you might have to include logic here that only samples a subset of the points if there are too many and it is slow. Maybe its fine though
            object_names = json.load(jf)
            classes = [obj.split('/')[0] for obj in object_names] 
            labels = [self.label_map[class_] for class_ in classes]

            data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
            labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices((data_tensor, labels_tensor))

            return dataset

    def load_data(self):
        filenames = self.get_filenames()
        combined_dataset = None


        for h5_file in filenames:
            name, _ = h5_file.split('.')
            json_file = name + "_id2name.json"

            dataset = self.dataset_from_h5(h5_file, json_file)
            if combined_dataset is None:
                combined_dataset = dataset
            else:
                combined_dataset = combined_dataset.concatenate(dataset)
        return combined_dataset

    ################################ Preprocessing ################################

    def preprocess(self, point_cloud, label):
        if self.normalize:
            point_cloud, label = self.add_normalization(point_cloud, label)
        if self.data_augmentation:
            point_cloud, label = self.add_rotation(point_cloud, label)
            point_cloud, label = self.add_jitter(point_cloud, label)
        return point_cloud, label

    def add_normalization(self, point_cloud, label):
        centroid = tf.reduce_mean(point_cloud, axis=0)
        point_cloud_centered = point_cloud - centroid
        max_dist = tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(point_cloud_centered), axis=1)))
        point_cloud_normalized = point_cloud_centered / max_dist
        return point_cloud_normalized, label

    def add_jitter(self, point_cloud, label):
        jitter = 0.001
        noise = tf.random.uniform(tf.shape(point_cloud), minval=-jitter, maxval=jitter)
        point_cloud += noise
        return point_cloud, label

    def add_rotation(self, point_cloud, label):
        theta = tf.random.uniform((), 0, 2 * tf.constant(np.pi))
        rotation_matrix = tf.reshape(tf.stack([tf.cos(theta),   0.0,    tf.sin(theta),
                                               0.0          ,   1.0,              0.0,
                                              -tf.sin(theta),   0.0,  tf.cos(theta)]), (3, 3))
        rotated_point_cloud = tf.linalg.matmul(point_cloud, rotation_matrix)
        return rotated_point_cloud, label
    
     ################################ Dataset Creation ################################
    def create_dataset(self):
        self.dataset = self.dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) 
        return self.dataset
