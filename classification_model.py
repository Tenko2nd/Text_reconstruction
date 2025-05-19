import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt # Assurez-vous d'importer matplotlib

# Couches d'augmentation de données
data_augmentation = Sequential(
    [
        layers.RandomRotation(0.1), # Rotation de +/- 10%
        layers.RandomZoom(0.1),     # Zoom de +/- 10%
        # layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Décalage léger
    ],
    name="data_augmentation",
)

class CustomModel(Model):
    def __init__(self, num_classes=94) -> None:
        super(CustomModel, self).__init__()

        self.augmentation = data_augmentation

        self.conv_block1 = Sequential([
            layers.Conv2D(32, (3, 3), padding='same', name="Conv1_1"),
            layers.BatchNormalization(name="BN1_1"),
            layers.Activation('relu', name="Relu1_1"),
            layers.Conv2D(32, (3, 3), padding='same', name="Conv1_2"),
            layers.BatchNormalization(name="BN1_2"),
            layers.Activation('relu', name="Relu1_2"),
            layers.MaxPooling2D((2, 2), name="MaxPool1")
        ])

        self.conv_block2 = Sequential([
            layers.Conv2D(64, (3, 3), padding='same', name="Conv2_1"),
            layers.BatchNormalization(name="BN2_1"),
            layers.Activation('relu', name="Relu2_1"),
            layers.Conv2D(64, (3, 3), padding='same', name="Conv2_2"),
            layers.BatchNormalization(name="BN2_2"),
            layers.Activation('relu', name="Relu2_2"),
            layers.MaxPooling2D((2, 2), name="MaxPool2")
        ])

        self.conv_block3 = Sequential([
            layers.Conv2D(128, (3, 3), padding='same', name="Conv3_1"),
            layers.BatchNormalization(name="BN3_1"),
            layers.Activation('relu', name="Relu3_1"),
            layers.Conv2D(128, (3, 3), padding='same', name="Conv3_2"),
            layers.BatchNormalization(name="BN3_2"),
            layers.Activation('relu', name="Relu3_2"),
            layers.MaxPooling2D((2, 2), name="MaxPool3")
        ])

        self.flatten = layers.Flatten()
        self.dense_block = Sequential([
            layers.Dense(256, name="Dense1"),
            layers.BatchNormalization(name="BNDense1"),
            layers.Activation('relu', name="ReluDense1"),
            layers.Dropout(0.5, name="Dropout1"),
            layers.Dense(num_classes, activation='softmax', name="OutputDense")
        ])

    def call(self, input_tensor, training=False) -> tf.Tensor:
        # La couche d'augmentation (Sequential) passera l'argument `training`
        # à ses sous-couches (RandomRotation, RandomZoom).
        # Si training=False, les transformations ne sont pas appliquées par ces couches.
        x = self.augmentation(input_tensor, training=training)
        
        x = self.conv_block1(x, training=training)
        x = self.conv_block2(x, training=training)
        x = self.conv_block3(x, training=training)
        x = self.flatten(x)
        x = self.dense_block(x, training=training)
        return x

def load_data_from_directory(data_dir: str, batch_size: int = 32, image_size=(28,28)):
    # Charger les images et les étiquettes à partir du répertoire
    initial_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode='int',
        shuffle=True,
        seed=123
    )
    
    # Récupérer class_names AVANT les transformations .map() et .prefetch()
    class_names = initial_dataset.class_names
    
    # Normalisation des données (mettre les pixels entre 0 et 1)
    normalization_layer = layers.Rescaling(1./255)
    
    # Appliquer la normalisation
    dataset = initial_dataset.map(lambda x, y: (normalization_layer(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE) # Ajout de num_parallel_calls

    # Optimisation des performances du pipeline de données
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, class_names # Retourner aussi class_names

if __name__ == '__main__':
    train_data_dir = "images"
    val_data_dir = "test"
    BATCH_SIZE = 64
    NUM_CLASSES = 94
    IMAGE_SIZE = (28,28)

    # Charger les données et récupérer class_names
    train_dataset, train_class_names = load_data_from_directory(train_data_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    val_dataset, val_class_names = load_data_from_directory(val_data_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    # Vérifier les noms des classes pour s'assurer qu'il y en a bien 94
    print(f"Nombre de classes détectées dans l'ensemble d'entraînement : {len(train_class_names)}")
    if len(train_class_names) != NUM_CLASSES:
        print(f"Attention: Le nombre de classes détectées dans l'entraînement ({len(train_class_names)}) ne correspond pas à NUM_CLASSES ({NUM_CLASSES})")
    # Optionnellement, vérifier aussi pour val_class_names si nécessaire

    model = CustomModel(num_classes=NUM_CLASSES)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(filepath='best_tmnist_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Construire le modèle
    # On peut utiliser compute_output_shape ou un batch de données réelles
    # Utiliser un batch de données est plus robuste
    # Il faut s'assurer que le dataset n'est pas vide
    # Note: le `take(1)` crée un nouveau dataset, il faut itérer dessus pour obtenir le batch
    for images, labels in train_dataset.take(1):
        if tf.shape(images)[0] > 0: 
            model(images) # Ici, training est False par défaut, donc l'augmentation ne s'appliquera pas
        break
    model.summary()

    print("\nDébut de l'entraînement...")
    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=val_dataset,
                        callbacks=callbacks)

    print("\nÉvaluation finale sur l'ensemble de validation ('test')...")
    loss, accuracy = model.evaluate(val_dataset, verbose=0) # verbose=0 pour éviter la barre de progression ici
    print(f"Perte (Loss) sur la validation : {loss:.4f}")
    print(f"Précision (Accuracy) sur la validation : {accuracy:.4f}")

    if history and history.history: # S'assurer que history et history.history ne sont pas None
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.legend()
            plt.title('Accuracy')
        else:
            print("Clés 'accuracy' ou 'val_accuracy' non trouvées dans history.history")


        plt.subplot(1, 2, 2)
        if 'loss' in history.history and 'val_loss' in history.history:
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.title('Loss')
        else:
            print("Clés 'loss' ou 'val_loss' non trouvées dans history.history")

        plt.show()