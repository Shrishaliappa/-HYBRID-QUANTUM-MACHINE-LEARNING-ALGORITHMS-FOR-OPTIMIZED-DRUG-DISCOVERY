from confusion_met import *
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models, Input
from tensorflow.keras import backend as K
import time
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def svm_ga(X_train, X_test, y_train, y_test):
    # Handle missing values (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Build SVM (RBF kernel as default)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

    # Train SVM
    model.fit(X_train_scaled, y_train)

    # Measure response latency (prediction time)
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    response_latency = (time.time() - start_time) * 1000  # in milliseconds
    # Evaluation
    met = multi_confu_matrix(y_test, y_pred)
    return met, response_latency

def build_cnn(input_dim, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim, 1)))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def D3cnn(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for Conv1D
    X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

    # Build and train CNN
    model = build_cnn(X_train.shape[1], num_classes)
    model.fit(X_train_cnn, y_train,
              epochs=100,
              batch_size=128,
              verbose=1,
              validation_split=0.2)

    # Predict with latency measurement
    start_time = time.time()
    y_probs = model.predict(X_test_cnn)
    response_latency = time.time() - start_time
    y_pred = np.argmax(y_probs, axis=1)
    # Evaluation
    met = multi_confu_matrix(y_test, y_pred)
    avg_latency_per_sample = response_latency / len(X_test)
    return met, response_latency


def cnn_model(xtrain, xtest, ytrain, ytest):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np

    # Label Encoding
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)

    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Feature Scaling
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Reshape for Conv1D
    xtrain_seq = np.expand_dims(xtrain_scaled, axis=2)  # (samples, features, 1)
    xtest_seq = np.expand_dims(xtest_scaled, axis=2)

    # CNN Model
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(xtrain_seq.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    # Compile & Train
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(xtrain_seq, ytrain_cat, epochs=20, batch_size=32,
              validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)

    # Predictions
    y_pred_probs = model.predict(xtest_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    met = multi_confu_matrix(ytest_encoded, y_pred)
    return y_pred, met
def CycleGAN(xtrain, xtest, ytrain, ytest):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # === Generator Block ===
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(3, 64, 7, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 7, padding=3),
                nn.Tanh()
            )

        def forward(self, x):
            return self.main(x)

    # === Discriminator Block ===
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 1, 4, padding=1)
            )

        def forward(self, x):
            return self.main(x)

    # === Initialize models ===
    G = Generator()  # X → Y
    F = Generator()  # Y → X
    DX = Discriminator()  # Real/fake X
    DY = Discriminator()  # Real/fake Y

    # === Optimizers ===
    g_opt = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=0.0002)
    d_opt = optim.Adam(list(DX.parameters()) + list(DY.parameters()), lr=0.0002)

    # === Loss Functions ===
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # === Convert numpy to torch ===
    xtrain = torch.FloatTensor(xtrain).permute(0, 3, 1, 2)  # (N, C, H, W)
    xtest = torch.FloatTensor(xtest).permute(0, 3, 1, 2)

    train_loader = DataLoader(xtrain, batch_size=4, shuffle=True)

    # === Training loop (simplified) ===
    for epoch in range(5):  # Just 5 epochs for simplicity
        for real_x in train_loader:
            # Simulated fake Y and back-translated X
            fake_y = G(real_x)
            recon_x = F(fake_y)

            # Cycle consistency loss
            cycle_loss = l1_loss(recon_x, real_x)

            # Generator adversarial losses (DY(fake_y) should be 1)
            g_loss_adv = mse_loss(DY(fake_y), torch.ones_like(DY(fake_y)))

            # Total generator loss
            g_loss = g_loss_adv + 10.0 * cycle_loss

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            # Train discriminators
            real_y = fake_y.detach()  # For demo: fake_y as real_y
            d_loss_real = mse_loss(DY(real_y), torch.ones_like(DY(real_y)))
            d_loss_fake = mse_loss(DY(G(real_x).detach()), torch.zeros_like(DY(G(real_x).detach())))
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

        print(f"Epoch {epoch+1}: G_loss = {g_loss.item():.4f}, D_loss = {d_loss.item():.4f}")

    # === Translate test images ===
    G.eval()
    xtest_fake_y = G(xtest)
    return xtest_fake_y
