import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def deep_learning(df):
    # Preprocess data and create feature vectors
    text_data = df['Key Skills']
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(text_data).toarray()

    # We want to predict the Role
    y = df['Role']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_text, y_encoded, test_size=0.2)

    # Build a neural network model
    model = Sequential()
    model.add(Input(shape=(X_text.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax')) # Output layer with softmax for multiclass classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    try:
        model.load_weights('./checkpoints/my_checkpoint')
    except:
        callbacks = [
            keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor="val_loss",
                # "no longer improving" being defined as "no better than 1e-2 less"
                min_delta=1e-2,
                verbose=1,
            )
        ]
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

        # Save the model
        model.save_weights('./checkpoints/my_checkpoint')
    
    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f'Test Accuracy: {accuracy}')
    
    # Obtain embeddings for all jobs in the training set
    embeddings_model = Sequential(model.layers[:-1])  # Remove the last dense layer
    job_embeddings = embeddings_model.predict(X_text)

    # Fit a KNN model on the embeddings
    knn_model = NearestNeighbors(n_neighbors=100, metric='cosine')  # You can adjust the number of neighbors and metric
    knn_model.fit(job_embeddings)

    
    return vectorizer, embeddings_model, knn_model


# def knn(df, n_neighbors):
#     # Preprocess data and create feature vectors
#     text_data = df['Key Skills']
#     vectorizer = TfidfVectorizer()
#     X_text = vectorizer.fit_transform(text_data).toarray()

#     # We want to predict the Role
#     y = df['Role']
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2)

#     # Build and train the model
#     knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
#     knn_model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = knn_model.predict(X_test)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy}')
    
#     return knn_model


# def get_recommendations(knn_model, vectorizer, skills):
#     # Find the 10 closest jobs to the new input based on cosine similarity
#     distances, indices = knn_model.kneighbors(skills)

#     # Print the closest jobs
#     for i, index in enumerate(indices[0]):
#         job_title = df.iloc[index]['Job Title']
#         role = df.iloc[index]['Role']
#         key_skills = df.iloc[index]['Key Skills']
#         distance = distances[0][i]
#         print(f"Rank {i + 1}: Role: {role}, Job Title: {job_title}, Distance: {distance}\nKey Skills: {key_skills}\n")
        
def get_recommendations_deep(df, knn_model, embeddings_model, vectorizer, skills, experience):
    skills = vectorizer.transform([skills]).toarray()
    skills_embedded = embeddings_model.predict(skills)
    
    # Find the 10 closest jobs to the new input based on cosine similarity
    distances, indices = knn_model.kneighbors(skills_embedded)
    
    # Print the closest jobs
    recommendations = ""
    n = 0
    for i, index in enumerate(indices[0]):
        experience_required = df.iloc[index]['Job Experience Required']
        print(experience_required)
        if experience_required == '' or experience >= int(experience_required):
            job_title = df.iloc[index]['Job Title']
            role = df.iloc[index]['Role']
            key_skills = df.iloc[index]['Key Skills']
            distance = distances[0][i]
            recommendations += f"Rank {n + 1} - Distance: {distance}\nRole: {role}\nJob Title: {job_title}\nMinimal experience required: {experience_required} year(s)\nKey Skills: {key_skills}\n\n"
            n += 1
        if n == 10:
            break
    return recommendations

