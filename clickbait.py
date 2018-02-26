from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with open("clickbait.txt") as f:
    lines = f.read().strip().split("\n")
    lines = [line.split("\t") for line in lines]
headlines, labels = zip(*lines)

# Split into training and testing groups
h_train, h_test, l_train, l_test = train_test_split(headlines, labels, test_size=.8)

# Create Vectorizer
# Scheme: Term-Frequency - Inverse Document Frequency (tf-idf)
# This can be used for text, to convert the information into vectors
vectorizer = TfidfVectorizer()

# Create Classifier
# Linear SVM Classifier
svm = LinearSVC()

# Vectorize the data.
# fit_transform assumes the vocabulary of the training data is our complete vocab
# transform will drop any words not already in our fit vocabulary of our training data
train_vectors = vectorizer.fit_transform(h_train)
test_vectors = vectorizer.transform(h_test)

# Train the classifier with the training vectors
svm.fit(train_vectors, l_train)

# Make predictions on test vectors
predictions = svm.predict(test_vectors)
display(h_test[0:5])
display(predictions[0:5])
display(l_test[0:5])

# Calculate accuracy
accuracy = accuracy_score(l_test, predictions)
print("Accuracy: ", accuracy)
