import ast
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

##############################################################################
# 1. CONFIGURATION & UTILS
##############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dataset_folder = "sp-25-tamu-csce-633-600-machine-learning"

def normalize_columns(df):
    # Strip whitespace, convert to lower-case, and replace spaces with underscores.
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def parse_cgm_data(cgm_str):
    # Parse the string representing a list of (timestamp, glucose_value) tuples.
    data = ast.literal_eval(cgm_str)
    parsed = []
    for (ts_str, glucose_val) in data:
        ts = pd.to_datetime(ts_str)
        parsed.append((ts, float(glucose_val)))
    return parsed

def summarize_cgm(cgm_list):
    # Compute summary statistics from the parsed CGM data.
    if len(cgm_list) == 0:
        return {"cgm_mean": 0, "cgm_min": 0, "cgm_max": 0, "cgm_std": 0, "cgm_count": 0}
    values = np.array([pt[1] for pt in cgm_list], dtype=float)
    return {
        "cgm_mean": float(values.mean()),
        "cgm_min": float(values.min()),
        "cgm_max": float(values.max()),
        "cgm_std": float(values.std(ddof=1)),
        "cgm_count": float(len(values))
    }

def parse_time_in_minutes(time_str):
    # Convert a time string (e.g., "9/19/2021 8:41") to minutes from midnight.
    dt = pd.to_datetime(time_str)
    return dt.hour * 60 + dt.minute

##############################################################################
# 2. LOADING & PARSING CGM DATA (TRAIN & TEST)
##############################################################################
def load_cgm_train():
    cgm_path = os.path.join(dataset_folder, "cgm_train.csv")
    df = pd.read_csv(cgm_path, encoding="utf-8-sig")
    print(df.columns.tolist())
    df = normalize_columns(df)  # Now columns like "subject_id", "day", "breakfast_time", "lunch_time", "cgm_data"
    print(df.columns.tolist())
    
    features = []
    for _, row in df.iterrows():
        subj_id = row["subject_id"]
        day = row["day"]
        breakfast_str = row["breakfast_time"]
        lunch_str = row["lunch_time"]
        cgm_str = row["cgm_data"]

        cgm_list = parse_cgm_data(cgm_str)
        cgm_stats = summarize_cgm(cgm_list)
        try:
            breakfast_min = parse_time_in_minutes(str(breakfast_str))
            lunch_min = parse_time_in_minutes(str(lunch_str))
            time_diff = lunch_min - breakfast_min
        except Exception:
            time_diff = 0.0

        features.append({
            "subject_id": subj_id,
            "day": day,
            **cgm_stats,
            "breakfast_lunch_diff": time_diff
        })
    return pd.DataFrame(features)

def load_cgm_test():
    cgm_path = os.path.join(dataset_folder, "cgm_test.csv")
    df = pd.read_csv(cgm_path)
    df = normalize_columns(df)
    
    features = []
    for _, row in df.iterrows():
        subj_id = row["subject_id"]
        day = row["day"]
        breakfast_str = row["breakfast_time"]
        lunch_str = row["lunch_time"]
        cgm_str = row["cgm_data"]

        cgm_list = parse_cgm_data(cgm_str)
        cgm_stats = summarize_cgm(cgm_list)
        try:
            breakfast_min = parse_time_in_minutes(str(breakfast_str))
            lunch_min = parse_time_in_minutes(str(lunch_str))
            time_diff = lunch_min - breakfast_min
        except Exception:
            time_diff = 0.0

        features.append({
            "subject_id": subj_id,
            "day": day,
            **cgm_stats,
            "breakfast_lunch_diff": time_diff
        })
    return pd.DataFrame(features)

##############################################################################
# 3. LOADING & PARSING DEMO/VIOME DATA (TRAIN & TEST)
##############################################################################
def parse_viome(viome_str):
    return [float(x) for x in viome_str.split(",")]

def load_demo_viome_train():
    demo_path = os.path.join(dataset_folder, "demo_viome_train.csv")
    df = pd.read_csv(demo_path)
    df = normalize_columns(df)
    if "viome" in df.columns:
        viome_expanded = df["viome"].apply(parse_viome)
        max_len = viome_expanded.apply(len).max()
        viome_data = pd.DataFrame(viome_expanded.to_list(), columns=[f"viome_{i}" for i in range(max_len)])
        df = pd.concat([df.drop(columns=["viome"]), viome_data], axis=1)
    if "race" in df.columns:
        df.drop(columns=["race"], inplace=True)
    return df

def load_demo_viome_test():
    demo_path = os.path.join(dataset_folder, "demo_viome_test.csv")
    df = pd.read_csv(demo_path)
    df = normalize_columns(df)
    if "viome" in df.columns:
        viome_expanded = df["viome"].apply(parse_viome)
        max_len = viome_expanded.apply(len).max()
        viome_data = pd.DataFrame(viome_expanded.to_list(), columns=[f"viome_{i}" for i in range(max_len)])
        df = pd.concat([df.drop(columns=["viome"]), viome_data], axis=1)
    if "race" in df.columns:
        df.drop(columns=["race"], inplace=True)
    return df

##############################################################################
# 4. LOADING & PARSING IMAGE DATA (TRAIN & TEST)
##############################################################################
def average_rgb(image_3d):
    pixels = []
    for row in image_3d:
        for pix in row:
            pixels.append(pix)
    arr = np.array(pixels, dtype=float)
    return float(arr[:,0].mean()), float(arr[:,1].mean()), float(arr[:,2].mean())

def parse_image_column(img_str):
    arr_3d = ast.literal_eval(img_str)
    return average_rgb(arr_3d)

def load_img_train():
    img_path = os.path.join(dataset_folder, "img_train.csv")
    df = pd.read_csv(img_path)
    df = normalize_columns(df)  # Expecting columns: "subject_id", "day", "image_before_breakfast", "image_before_lunch"
    features = []
    for _, row in df.iterrows():
        subj_id = row["subject_id"]
        day = row["day"]
        br_str = row["image_before_breakfast"]
        ln_str = row["image_before_lunch"]
        br_rgb = parse_image_column(br_str)
        ln_rgb = parse_image_column(ln_str)
        features.append({
            "subject_id": subj_id,
            "day": day,
            "img_bf_breakfast_r": br_rgb[0],
            "img_bf_breakfast_g": br_rgb[1],
            "img_bf_breakfast_b": br_rgb[2],
            "img_bf_lunch_r": ln_rgb[0],
            "img_bf_lunch_g": ln_rgb[1],
            "img_bf_lunch_b": ln_rgb[2]
        })
    return pd.DataFrame(features)

def load_img_test():
    img_path = os.path.join(dataset_folder, "img_test.csv")
    df = pd.read_csv(img_path)
    df = normalize_columns(df)
    features = []
    for _, row in df.iterrows():
        subj_id = row["subject_id"]
        day = row["day"]
        br_str = row["image_before_breakfast"]
        ln_str = row["image_before_lunch"]
        br_rgb = parse_image_column(br_str)
        ln_rgb = parse_image_column(ln_str)
        features.append({
            "subject_id": subj_id,
            "day": day,
            "img_bf_breakfast_r": br_rgb[0],
            "img_bf_breakfast_g": br_rgb[1],
            "img_bf_breakfast_b": br_rgb[2],
            "img_bf_lunch_r": ln_rgb[0],
            "img_bf_lunch_g": ln_rgb[1],
            "img_bf_lunch_b": ln_rgb[2]
        })
    return pd.DataFrame(features)

##############################################################################
# 5. LOADING LABELS
##############################################################################
def load_label_train():
    label_path = os.path.join(dataset_folder, "label_train.csv")
    df = pd.read_csv(label_path, sep="\t")
    df = normalize_columns(df)  # Expecting: "subject_id", "day", "calories_total"
    return df

##############################################################################
# 6. MERGING TRAINING & TEST DATASETS
##############################################################################
def build_train_data():
    cgm_df = load_cgm_train()
    demo_df = load_demo_viome_train()
    img_df = load_img_train()
    label_df = load_label_train()

    # Merge on "subject_id" and "day" for CGM, image, and label data.
    merged = pd.merge(cgm_df, img_df, on=["subject_id", "day"], how="left")
    merged = pd.merge(merged, label_df, on=["subject_id", "day"], how="left")
    # Merge demo (which may only have subject_id) on "subject_id"
    merged = pd.merge(merged, demo_df, on=["subject_id"], how="left")
    return merged

def build_test_data():
    cgm_df = load_cgm_test()
    demo_df = load_demo_viome_test()
    img_df = load_img_test()

    merged = pd.merge(cgm_df, img_df, on=["subject_id", "day"], how="left")
    merged = pd.merge(merged, demo_df, on=["subject_id"], how="left")
    return merged

##############################################################################
# 7. BASELINE MODEL & TRAINING
##############################################################################
class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim):
        super(SimpleFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def main():
    print("Building training data ...")
    train_df = build_train_data()
    print("Train shape:", train_df.shape)
    
    # Use "subject_id" and "day" as keys; target is "calories_total"
    key_cols = ["subject_id", "day"]
    target_col = "calories_total"
    feature_cols = [c for c in train_df.columns if c not in key_cols + [target_col]]
    print("Feature columns:", feature_cols)
    
    # Drop rows with missing target
    train_df = train_df.dropna(subset=[target_col])
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
    
    model = SimpleFeedForward(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train_t)
        loss_train = criterion(pred_train, y_train_t)
        loss_train.backward()
        optimizer.step()
    
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            loss_val = criterion(pred_val, y_val_t)
    
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")
    
    print("Building test data ...")
    test_df = build_test_data()
    print("Test shape:", test_df.shape)
    # Use same feature columns
    test_features = [c for c in feature_cols if c in test_df.columns]
    test_df[test_features] = test_df[test_features].fillna(0)
    X_test = test_df[test_features].values
    X_test = scaler.transform(X_test)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test_t).cpu().numpy().flatten()
    
    submission = pd.DataFrame({
        "row_id": np.arange(len(preds_test)),
        "label": preds_test
    })
    submission.to_csv("submission.csv", index=False)
    print("submission.csv saved successfully.")

if __name__ == "__main__":
    main()
