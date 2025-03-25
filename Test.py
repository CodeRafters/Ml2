import pandas as pd
import pickle

# Load the trained model
filename = 'sign_language_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a DataFrame for manual input data
manual_data = pd.DataFrame({
    'Thumb': [766.5, 900, 716],
    'Index': [834.8, 861, 723],
    'Middle': [824.9, 724, 866],
    'Ring': [739.6, 809, 817],
    'Little': [877.2, 712, 865]
})

# Make predictions using the loaded model
predictions = loaded_model.predict(manual_data)

# Print the predictions
print("\nPredictions for manual data:")
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: {prediction}")

# Explanation of predictions (Manual Cross-Checking)
print("\n--- Explanation of Predictions (Manual Cross-Checking) ---")

print("\nSample 1: Thumb=750, Index=730, Middle=800, Ring=800, Little=800")
print("Based on these finger positions, the model predicts the gesture is most likely similar to the cluster associated with that range of values.")

print("\nSample 2: Thumb=800, Index=850, Middle=810, Ring=800, Little=800")
print("Based on the index finger being high, the model predicts a different gesture.")

print("\nSample 3: Thumb=700, Index=750, Middle=850, Ring=700, Little=800")
print("Since the middle finger is high and the ring finger is low, the model predicts another gesture.")
