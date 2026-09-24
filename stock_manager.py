import pandas as pd
import os

STOCK_FILE = "stock_data.csv"
REORDER_FILE = "reorder_list.csv"

LOW_STOCK_THRESHOLD = 10


def load_stock():
    return pd.read_csv(STOCK_FILE)


def check_availability(medicine_name, stock_df):
    row = stock_df[stock_df["MedicineName"] == medicine_name]
    if row.empty:
        return None
    quantity = int(row.iloc[0]["Quantity"])
    if quantity == 0:
        status = "Out of stock"
    elif quantity <= LOW_STOCK_THRESHOLD:
        status = "Low stock"
    else:
        status = "In stock"
    return {"name": medicine_name, "quantity": quantity, "status": status}


def add_to_reorder_list(medicine_name, quantity_needed=1):
    if os.path.exists(REORDER_FILE):
        reorder_df = pd.read_csv(REORDER_FILE)
    else:
        reorder_df = pd.DataFrame(columns=["MedicineName", "QuantityNeeded"])

    if medicine_name in reorder_df["MedicineName"].values:
        return f"{medicine_name} is already on the reorder list."

    new_row = pd.DataFrame([{"MedicineName": medicine_name, "QuantityNeeded": quantity_needed}])
    reorder_df = pd.concat([reorder_df, new_row], ignore_index=True)
    reorder_df.to_csv(REORDER_FILE, index=False)
    return f"Added {medicine_name} to reorder list."


def get_reorder_list():
    if os.path.exists(REORDER_FILE):
        return pd.read_csv(REORDER_FILE)
    return pd.DataFrame(columns=["MedicineName", "QuantityNeeded"])


if __name__ == "__main__":
    stock_df = load_stock()

    # test with a known low-stock item
    result = check_availability("Alprax 0-25", stock_df)
    print(result)

    # test with an out-of-stock item
    result2 = check_availability("Pacimol650", stock_df)
    print(result2)

    # test adding to reorder list
    print(add_to_reorder_list("Pacimol650"))
    print(add_to_reorder_list("Alprax 0-25"))
    print(add_to_reorder_list("Pacimol650"))  # should say already on list

    print("\nCurrent reorder list:")
    print(get_reorder_list())