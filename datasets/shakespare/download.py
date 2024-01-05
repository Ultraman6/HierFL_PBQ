import requests
from tqdm import tqdm
import os


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_file_from_google_drive(file_id, destination_folder, file_name):
    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 32768  # Size of each download chunk in bytes

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with requests.Session() as session:
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        file_path = os.path.join(destination_folder, file_name)
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, "wb") as file, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    size = file.write(chunk)
                    bar.update(size)


def download_shakespeare_dataset(base_path):
    """
    Downloads the Shakespeare dataset for training and testing to a specified base path.

    :param base_path: Base directory to save the Shakespeare dataset.
    """
    # Ensure the base path exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    # Google Drive file IDs for the Shakespeare dataset
    train_file_id = '1mD6_4ju7n2WFAahMKDtozaGxUASaHAPH'
    test_file_id = '1GERQ9qEJjXk_0FXnw1JbjuGCI-zmmfsk'

    # Download train and test datasets
    download_file_from_google_drive(train_file_id, train_path, 'all_data_niid_2_keep_0_train_8.json')
    download_file_from_google_drive(test_file_id, test_path, 'all_data_niid_2_keep_0_test_8.json')

    print('Shakespeare dataset downloaded successfully to:', base_path)


# Example usage
if __name__ == "__main__":
    base_path = '~/shakespeare'  # Replace with your desired path
    download_shakespeare_dataset(base_path)