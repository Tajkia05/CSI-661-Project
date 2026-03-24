# ============================================================
# Person 1 - Data + Analysis
# MovieLens 1M Behavioral Baseline for Gender Prediction
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---------------- LOAD DATA ----------------
ratings = pd.read_csv(
    'data/ratings.dat',
    sep='::',
    engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

users = pd.read_csv(
    'data/users.dat',
    sep='::',
    engine='python',
    names=['user_id', 'gender', 'age', 'occupation', 'zip']
)


# ---------------- MERGE & PREPARE DATA ----------------
df = ratings.merge(users, on='user_id')
df = df[['user_id', 'movie_id', 'rating', 'gender', 'age']]

print("\n--- Dataset Preview ---")
print(df.head())
print("Shape:", df.shape)

print("\n--- Data Integrity Checks ---")
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Ensure valid rating range
df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]


# ---------------- EDA ----------------

# Gender distribution (user-level)
plt.figure()
users['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution (User Level)')
plt.ylabel('Count')
plt.savefig('gender_distribution.png')
plt.close()

# Age distribution (user-level)
plt.figure()
users['age'].value_counts().sort_index().plot(kind='bar')
plt.title('Age Distribution')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.close()

# Ratings per user (user activity)
ratings_per_user = df.groupby('user_id').size()

plt.figure()
ratings_per_user.hist(bins=50)
plt.title('Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.savefig('ratings_per_user.png')
plt.close()

print("\nEDA plots saved.")


# ---------------- FEATURE ENGINEERING ----------------

# Convert rating-level -> user-level
user_features = df.groupby('user_id').agg({
    'rating': ['count', 'mean', 'std']
}).reset_index()

user_features.columns = ['user_id', 'num_ratings', 'avg_rating', 'rating_std']
user_features['rating_std'] = user_features['rating_std'].fillna(0)

# Add gender label per user
user_labels = users[['user_id', 'gender']].drop_duplicates()
data = user_features.merge(user_labels, on='user_id')

# Encode gender
data['gender'] = data['gender'].map({'M': 1, 'F': 0})

print("\n--- User-Level Class Balance ---")
print(data['gender'].value_counts(normalize=True))


# ---------------- PREPARE DATA ----------------

X = data[['num_ratings', 'avg_rating', 'rating_std']]
y = data['gender']


# ---------------- TRAIN / TEST SPLIT ----------------
# Split by user_id so this baseline can use the same held-out users
# as the MF attacker model

train_users, test_users = train_test_split(
    data['user_id'],
    test_size=0.2,
    random_state=42,
    stratify=data['gender']
)

train_mask = data['user_id'].isin(train_users)
test_mask = data['user_id'].isin(test_users)

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]


# ---------------- MODEL ----------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ---------------- EVALUATION ----------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Majority baseline
majority = y_train.mode()[0]
base_acc = accuracy_score(y_test, [majority] * len(y_test))

print("\n--- Results ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Majority Accuracy: {base_acc:.4f}")


# ---------------- INTERPRETATION ----------------

print("\n--- Interpretation ---")

if acc <= base_acc:
    print("Behavioral baseline does NOT outperform majority baseline.")
else:
    print("Behavioral baseline improves over majority baseline.")

print("Accuracy should be interpreted carefully because the dataset is imbalanced.")
print("AUC near 0.5 indicates weak predictive power.")
print("Conclusion: Simple behavioral features are not sufficient to predict gender.")
