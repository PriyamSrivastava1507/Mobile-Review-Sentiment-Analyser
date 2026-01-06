import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("content/mobile-reviews.csv")

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["review_text"] = df["review_text"].apply(preprocess)

df["sentiment"] = df["sentiment"].map({
    "Positive": 1,
    "Negative": 0
})

df = df.dropna(subset=["review_text", "sentiment"])

sns.countplot(data=df, x="sentiment")
plt.xticks([0, 1], ["Negative", "Positive"])
plt.savefig("result_fig/review_stats.png")
plt.show()

X = df[["review_text", "helpful_votes", "review_length", "verified_purchase"]]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=7000, ngram_range=(1, 2), min_df=5), "review_text"),
        ("num", Pipeline([("scaler", StandardScaler())]), ["helpful_votes", "review_length"]),
        ("bin", "passthrough", ["verified_purchase"])
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.savefig("result_fig/confusion_matrix.png")
plt.show()

error_df = X_test.copy()
error_df["actual"] = y_test.values
error_df["predicted"] = y_pred

misclassified = error_df[error_df["actual"] != error_df["predicted"]]

print(len(misclassified))
print(misclassified[["review_text", "actual", "predicted"]].head())