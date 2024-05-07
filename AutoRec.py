from time import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

before = time()

# Load business data
business = pd.read_json('yelp_academic_dataset_business.json', lines = True)

# Load review data
review = pd.read_json('yelp_academic_dataset_review.json', lines = True)

# Load user data
user = pd.read_json('yelp_academic_dataset_user.json', lines = True)

# Set the business numbers to each unique business IDs
if 'business_no' not in business.columns and business['business_id'].nunique() == len(business['business_id']):
    business.insert(0,'business_no', range(len(business)), True)

# Set the user numbers to each unique user IDs
user_id = pd.DataFrame({'user_no': np.arange(len(review['user_id'].unique())),
                        'user_id': review['user_id'].unique()})

if 'user_no' not in user.columns:
    user.insert(0,'user_no', None)

# Create a mapping between user_id and user_no from the 'user_id'
user_dict = dict(zip(user_id['user_id'], user_id['user_no']))
# Use map to fill in the user_no in the 'user'
user['user_no'] = user['user_id'].map(user_dict)

# Map the business numbers and user numbers according to the user IDs and business IDs in the 'review' dataframe

# Create new coloumns in 'review' to be mapped into
if 'business_no' not in review.columns:
    review.insert(0,'business_no', None)
if 'user_no' not in review.columns:
    review.insert(0,'user_no', None)

# Create a mapping between business_id and business_no from the 'business'
business_dict = dict(zip(business['business_id'], business['business_no']))
# Use map to fill in the missing business_no in the 'review'
review['business_no'] = review['business_id'].map(business_dict)

# Create a mapping between user_id and user_no from the 'user_id'
user_dict = dict(zip(user_id['user_id'], user_id['user_no']))
# Use map to fill in the missing business_no in the 'review'
review['user_no'] = review['user_id'].map(user_dict)

# Extract the user_no, business_no, and ratings only
ratings_table = review[['user_no', 'business_no', 'stars']]

# Remove the duplicated rows and check.
row_to_drop = 164117
if ratings_table['user_no'].iloc[row_to_drop]==848 and ratings_table['business_no'].iloc[row_to_drop]==9423:
    ratings_table = ratings_table.drop(row_to_drop).reset_index(drop=True)

ratings_table['stars'] /= 5

# Check if GPU is available
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU
    device = torch.device("cuda")
    print("GPU is available. Using GPU.")
else:
    # If GPU is not available, use CPU
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Splitting the the data (table) into train/dev/test (70/10/20) sets
temp_ratings_table, test_ratings_table = train_test_split(ratings_table, test_size=0.2, random_state=42)
train_ratings_table, validation_ratings_table = train_test_split(temp_ratings_table, test_size=0.125, random_state=42)

# Define the rows and columns first
# Rows
num_users = len(ratings_table['user_no'].unique())
# Columns
num_items = len(ratings_table['business_no'].unique())

# Construct the whole matrix
# Use csr_array to store the sparse matrix more efficiently
ratings = np.array(ratings_table['stars'])
row = np.array(ratings_table['user_no'])
column = np.array(ratings_table['business_no'])
# Map the data into the csr array
ratings_matrix = csr_array((ratings, (row, column)), shape=(num_users,num_items), dtype=np.float32).toarray()

# Contruct the train matrix
# Define the ratings, row, column in train ratings table
ratings = np.array(train_ratings_table['stars'])
row = np.array(train_ratings_table['user_no'])
column = np.array(train_ratings_table['business_no'])
# Map the data into the csr array
train_matrix = csr_array((ratings, (row, column)), shape=(num_users,num_items), dtype=np.float32).toarray()

# Construct the validation matrix
# Define the ratings, row, column in validation ratings table
ratings = np.array(validation_ratings_table['stars'])
row = np.array(validation_ratings_table['user_no'])
column = np.array(validation_ratings_table['business_no'])
# Map the data into the csr array
validation_matrix = csr_array((ratings, (row, column)), shape=(num_users,num_items), dtype=np.float32).toarray()

# Construct the test matrix
# Define the ratings, row, column in test ratings table
ratings = np.array(test_ratings_table['stars'])
row = np.array(test_ratings_table['user_no'])
column = np.array(test_ratings_table['business_no'])
# Map the data into the csr array
test_matrix = csr_array((ratings, (row, column)), shape=(num_users,num_items), dtype=np.float32).toarray()

# Convert data to PyTorch tensors
train_tensor = torch.FloatTensor(train_matrix)
validation_tensor = torch.FloatTensor(validation_matrix)
test_tensor = torch.FloatTensor(test_matrix)

# Create DataLoader
bs = 16
train_loader = DataLoader(train_tensor, batch_size=bs, shuffle=True)
validation_loader = DataLoader(validation_tensor, batch_size=bs, shuffle=False)
test_loader = DataLoader(test_tensor, batch_size=bs, shuffle=False)

# Define the Autoencoder model
class AutoRec(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.05):
        super(AutoRec, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Sigmoid activation to constrain outputs to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

version = f'BS{bs}_v2'
files_to_zip = []

# Define the parameters
input_size = num_items
hidden_size = [512,1024,2048,4096]
learning_rate = [0.00001,0.0001,0.001,0.01]
dropout = 0.05

epochs = 100
train_RMSE_all = []
validation_RMSE_all = []

# Trainings & Validations
for lr in learning_rate:
    print(f'Learning Rate: {lr}')
    for size in hidden_size:
        print(f'Hidden Size: {size}')

        # Define the model
        model = AutoRec(input_size, size, dropout).to(device)
        # Define the loss
        criterion = nn.MSELoss()
        # Define the optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        train_RMSE = []
        validation_RMSE = []
        for epoch in range(epochs):
            train_loss = 0.0
            total_non_zero_elements = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon = model(data)
                non_zero_mask = (data != 0)
                # Mask and extract non-zero elements from both reconstructed and actual data
                recon_non_zero = recon[non_zero_mask]
                data_non_zero = data[non_zero_mask]
                if data_non_zero.numel() > 0:
                    loss = criterion(recon_non_zero, data_non_zero)
                    loss.backward()
                    optimizer.step()
                    train_loss += np.sqrt(loss.item()) * data_non_zero.size(0)
                    total_non_zero_elements += non_zero_mask.sum().item()
            train_loss /= total_non_zero_elements
            train_loss *= 5
            train_RMSE.append(train_loss)

            validation_loss = 0.0
            total_non_zero_elements = 0
            with torch.no_grad():
                for data in validation_loader:
                    data = data.to(device)
                    recon = model(data)
                    non_zero_mask = (data != 0)
                    # Mask and extract non-zero elements from both reconstructed and actual data
                    recon_non_zero = recon[non_zero_mask]
                    data_non_zero = data[non_zero_mask]
                    if data_non_zero.numel() > 0:
                        loss = criterion(recon_non_zero, data_non_zero)
                        validation_loss += np.sqrt(loss.item()) * data_non_zero.size(0)
                        total_non_zero_elements += non_zero_mask.sum().item()
            validation_loss /= total_non_zero_elements
            validation_loss *= 5
            validation_RMSE.append(validation_loss)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}')

        train_RMSE_all.append(train_RMSE)
        validation_RMSE_all.append(validation_RMSE)

        # Plot Each Combination
        plt.plot(range(1,epochs+1), train_RMSE, label='Train RMSE', color='blue')
        plt.plot(range(1,epochs+1), validation_RMSE, label='Validation RMSE', linestyle = '--',  color='purple')

        plt.xlim(0,epochs+1)
        plt.ylim(0,2.0)
        plt.xticks(np.arange(0,epochs+1,10))
        plt.yticks(np.arange(0,2.1,0.1))
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.title(f'RMSE of Hidden Size {size} with Learning Rate {lr}')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'AE_HS{size}_LR{lr}_{version}.png')
        files_to_zip.append(f'AE_HS{size}_LR{lr}_{version}.png')
        plt.show()
        plt.close()

        # Save the model after tuning
        file_name = f"AE_HS{size}_LR{lr}_{version}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, file_name)
        files_to_zip.append(file_name)
        print(f"Model saved to {file_name}")

        now = time()
        hours = (now-before)/3600
        print(f'{hours} hours')

# Plot each hidden size for each learning rate
for j, lr in enumerate(learning_rate):
    # Define the color palette based on the number of hidden_size
    palette = sns.color_palette("husl", len(hidden_size))
    plt.figure(figsize=(10,6))

    # Plot RMSE
    for i, size in enumerate(hidden_size):
        plt.plot(range(1,epochs+1), train_RMSE_all[i+len(hidden_size)*j], label=f'Hidden Size {size} (Train)', color=palette[i])
        plt.plot(range(1,epochs+1), validation_RMSE_all[i+len(hidden_size)*j], label=f'Hidden Size {size} (Validation)', linestyle='--', color=palette[i])
    plt.xlim(0,epochs+1)
    plt.ylim(0,2.0)
    plt.xticks(np.arange(0,epochs+1,10))
    plt.yticks(np.arange(0,2.1,0.1))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.title(f'RMSE with Learning Rate {lr}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'AE_RMSE_LR{lr}_{version}.png')
    files_to_zip.append(f'AE_RMSE_LR{lr}_{version}.png')
    plt.show()
    plt.close()

# Plot each learning rate for each hidden size
for j, size in enumerate(hidden_size):
    # Define the color palette based on the number of hidden_size
    palette = sns.color_palette("husl", len(learning_rate))
    plt.figure(figsize=(10,6))

    # Plot RMSE
    for i, lr in enumerate(learning_rate):
        plt.plot(range(1,epochs+1), train_RMSE_all[j+len(learning_rate)*i], label=f'Learning Rate {lr} (Train)', color=palette[i])
        plt.plot(range(1,epochs+1), validation_RMSE_all[j+len(learning_rate)*i], label=f'Learning Rate {lr} (Validation)', linestyle='--', color=palette[i])
    plt.xlim(0,epochs+1)
    plt.ylim(0,2.0)
    plt.xticks(np.arange(0,epochs+1,10))
    plt.yticks(np.arange(0,2.1,0.1))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.title(f'RMSE with Hidden Size {size}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'AE_RMSE_HS{size}_{version}.png')
    files_to_zip.append(f'AE_RMSE_HS{size}_{version}.png')
    plt.show()
    plt.close()

# Plot all
plt.figure(figsize=(16,10))
for j, lr in enumerate(learning_rate):
    # Plot RMSE
    for i, size in enumerate(hidden_size):
        # Define the color palette based on the number of hidden_size
        palette = sns.color_palette("husl", len(learning_rate)*len(hidden_size))
        plt.plot(range(1,epochs+1), train_RMSE_all[i+len(hidden_size)*j], label=f'Learning Rate {lr}, Hidden Size {size} (Train)', color=palette[i+len(hidden_size)*j])
        plt.plot(range(1,epochs+1), validation_RMSE_all[i+len(hidden_size)*j], label=f'Learning Rate {lr}, Hidden Size {size} (Validation)', linestyle='--', color=palette[i+len(hidden_size)*j])

plt.xlim(0,epochs+1)
plt.ylim(0,2.0)
plt.xticks(np.arange(0,epochs+1,10))
plt.yticks(np.arange(0,2.1,0.1))
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.title('All Train RMSE')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'AE_RMSE_All_{version}.png')
files_to_zip.append(f'AE_RMSE_All_{version}.png')
plt.show()
plt.close()

# Create a zip file object in write mode
with zipfile.ZipFile(f'{version}.zip', 'w') as zipf:
    # Loop through the list of files
    for file_to_zip in files_to_zip:
        # Add each file to the zip archive
        zipf.write(file_to_zip)

print(f'Files have been saved to {version}.zip')

# Below may or may not run in gpu
test_loss = 0.0
total_non_zero_elements = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        recon = model(data)
        non_zero_mask = (data != 0)
        # Mask and extract non-zero elements from both reconstructed and actual data
        recon_non_zero = recon[non_zero_mask]
        data_non_zero = data[non_zero_mask]
        if data_non_zero.numel() > 0:
            loss = criterion(recon_non_zero, data_non_zero)
            test_loss += np.sqrt(loss.item()) * data_non_zero.size(0)
            total_non_zero_elements += non_zero_mask.sum().item()
test_loss /= total_non_zero_elements
test_loss *= 5
print(f"RMSE for non-zero elements: {test_loss:.4f}")

def recommend_user(user_no, top_n=10):
    user_data = torch.tensor(ratings_matrix[user_no], dtype=torch.float32).to(device)
    encoded_data = model(user_data).cpu().detach().numpy()
    top_n_business_no = np.argsort(encoded_data)[::-1][:top_n]
    top_n_business = []
    for business_no in top_n_business_no:
        top_n_business.append(business[business['business_no']==business_no]['name'].item())

    top_n_business_ratings = np.sort(encoded_data)[::-1][:top_n]*5

    top_n_recommendations = pd.DataFrame({'Top': range(1,top_n+1), 'Business Name': top_n_business, 'Predicted Rating': top_n_business_ratings})

    return top_n_recommendations

# Example recommendation for user
user_no = 429
top_n = 10
recommended_businesses = recommend_user(user_no, top_n)
name = user[user['user_no']==user_no]['name'].item()
print(f'Recommended businesses for {name}:')
print(recommended_businesses.to_string(index = False))

now = time()
hours = (now-before)/3600
print(f'{hours} hours')
