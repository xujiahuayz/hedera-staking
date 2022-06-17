from staking.utils import fetch_save_data

if __name__ == "__main__":
    fetch_save_data(
        limit=100,
        order="asc",
        file_name="transactions",
        q="transactions",
        number_iterations=50,
    )

# INFO:root:No.0====/api/v1/transactions?limit=100&order=asc&
# timestamp=gt:1568412964.375367001
