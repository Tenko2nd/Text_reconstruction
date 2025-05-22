import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CustomModel(Model):

    def __init__(self) -> None:
        super(CustomModel, self).__init__()

        self.first_cnn_layer = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="First_CNN_Layer")
        self.first_pooling = layers.MaxPooling2D((2, 2), name="First_MaxPool_Layer")

        self.second_cnn_layer = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="Second_CNN_Layer")
        self.second_pooling = layers.MaxPooling2D((2, 2), name="Second_MaxPool_Layer")

        self.third_cnn_layer = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="Third_CNN_Layer")
        self.third_pooling = layers.MaxPooling2D((2, 2), name="Third_MaxPool_Layer")

        self.flatten = layers.Flatten()

        self.first_dense_layer = layers.Dense(64, activation='relu', name="First_Dense_Layer")
        self.second_dense_layer = layers.Dense(94, activation='softmax', name="Second_Dense_Layer")  # 94 classes for TMNIST

    def call(self, input_tensor, training=False) -> tf.Tensor:
        x = self.first_cnn_layer(input_tensor, training=training)
        x = self.first_pooling(x)
        x = self.second_cnn_layer(x, training=training)
        x = self.second_pooling(x)
        x = self.third_cnn_layer(x, training=training)
        x = self.third_pooling(x)
        x = self.flatten(x)
        x = self.first_dense_layer(x, training=training)
        x = self.second_dense_layer(x, training=training)

        return x


def load_data_from_directory(data_dir: str, batch_size: int = 32, augment: bool = False):
    
    # Charger les images et les étiquettes à partir du répertoire
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(28, 28),  # Les images sont déjà de taille 28x28
        batch_size=batch_size,
        color_mode="grayscale",  # Les images sont en niveaux de gris
        label_mode='int',  # Utilisez des indices d'entiers pour les labels (pas de one-hot encoding)
        shuffle=True,  # Mélanger les données
        seed=123  # Pour la reproductibilité
    )
    
    # Normalisation des données (mettre les pixels entre 0 et 1)
    normalization_layer = layers.Rescaling(1./255)

    data_augmentation_layers = tf.keras.Sequential([
            layers.RandomRotation(0.1), # Rotation de +/- 10% (environ 36 degrés)
            layers.RandomZoom(0.1),     # Zoom de +/- 10%
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Décalage de +/- 10%
        ])

    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation_layers(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        
    # Optimisation du pipeline de données
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    train_data_dir = "images"  # Remplacez par le chemin vers votre dossier "images"
    val_data_dir = "test"
    BATCH_SIZE = 32

    # Charger les données
    train_dataset = load_data_from_directory(train_data_dir, batch_size=BATCH_SIZE, augment=True)
    val_dataset = load_data_from_directory(val_data_dir, batch_size=BATCH_SIZE, augment=False)

    # Initialiser le modèle
    model = CustomModel()

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Utilisez 'sparse_categorical_crossentropy' pour des labels entiers
                  metrics=['accuracy'])

    callbacks_list = [
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001),
    ModelCheckpoint(filepath='best_model_tmnist.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
                  

    # Entraîner le modèle
    model.fit(train_dataset, 
            epochs=50, 
            validation_data=val_dataset,
            callbacks=callbacks_list)

    # Résumé du modèle
    model.summary()


    # Évaluation finale sur l'ensemble de validation ('test' dans votre cas)
    print("\nÉvaluation finale sur l'ensemble de validation ('test')...")
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Perte (Loss) sur la validation : {loss:.4f}")
    print(f"Précision (Accuracy) sur la validation : {accuracy:.4f}")

    model.save("model_classification_alphabet") 
    print("Modèle enregistré au format SavedModel dans le répertoire 'model_classification_alphabet'")