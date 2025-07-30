import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def brownian_cheetah_feature_selection(X, y, n_agents=10, max_iter=20, Q=0.5):
    n_features = X.shape[1]
    agents = np.random.randint(0, 2, size=(n_agents, n_features))  # Each agent is a binary vector
    best_score = -np.inf
    best_agent = None

    for iteration in range(max_iter):
        for i in range(n_agents):
            mask = agents[i].astype(bool)
            if np.sum(mask) == 0:
                continue  # Skip empty selections

            X_selected = X.iloc[:, mask]
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            score = np.mean(cross_val_score(clf, X_selected, y, cv=3))

            # === Selection (Equation 12) ===
            if score > best_score:
                best_score = score
                best_agent = agents[i].copy()

        # === Brownian Cheetah Movement: Attack + Exploration + Brownian Enhancement ===
        for i in range(n_agents):
            random_cheetah = agents[np.random.randint(n_agents)]
            delta = random_cheetah - agents[i]  # Distance between two cheetahs (Eq. 11)

            # Brownian motion (Eq. 13)
            brownian = np.random.randn(n_features)  # randn ~ N(0, 1)
            CF = ((max_iter - iteration) / max_iter) ** 2  # Convergence factor (Eq. 14)
            gamma = 0.001 * max_iter / (iteration + 1)     # Adaptive gamma (Eq. 6)

            # Motion step with BMS (Eq. 13)
            new_position = agents[i] + Q * CF * delta + gamma * brownian

            # Convert to probability and then binary mask
            agents[i] = (1 / (1 + np.exp(-new_position))) > 0.5  # Sigmoid-based binarization

    # Return final selected features
    selected_features = X.columns[best_agent.astype(bool)].tolist()
    return selected_features
