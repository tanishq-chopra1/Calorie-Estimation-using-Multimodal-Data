# test1.py
import ast, os
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. CONFIGURATION
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "sp-25-tamu-csce-633-600-machine-learning"
CGM_SEQ_LEN = 100
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# 2. UTILITIES
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def parse_cgm(s):
    try:
        arr = ast.literal_eval(s)
        return [float(v) for _, v in arr]
    except:
        return []

def pad_trunc(seq, L=CGM_SEQ_LEN):
    return seq[:L] + [0.0] * max(0, L - len(seq))

def parse_img(s):
    try:
        img = np.array(ast.literal_eval(s), dtype=np.float32)  # H×W×3
        t = torch.tensor(img).permute(2,0,1).unsqueeze(0)      # 1×3×H×W
        t = F.interpolate(t, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False)
        return t.squeeze(0)  # 3×IMG_SIZE×IMG_SIZE
    except:
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)

# 3. DATASET
class MultiModalDataset(Dataset):
    def __init__(self, split="train", scaler=None):
        cgm = pd.read_csv(os.path.join(DATA_DIR, f"cgm_{split}.csv"), encoding="utf-8-sig")
        demo= pd.read_csv(os.path.join(DATA_DIR, f"demo_viome_{split}.csv"), encoding="utf-8-sig")
        img = pd.read_csv(os.path.join(DATA_DIR, f"img_{split}.csv"), encoding="utf-8-sig")
        normalize_cols(cgm); normalize_cols(demo); normalize_cols(img)
        if split=="train":
            lbl = pd.read_csv(os.path.join(DATA_DIR, "label_train.csv"), encoding="utf-8-sig")
            normalize_cols(lbl)
        # expand viome
        if "viome" in demo:
            v = demo["viome"].apply(lambda s: [float(x) for x in s.split(",")] if isinstance(s,str) else [])
            L = v.apply(len).max()
            vdf = pd.DataFrame(v.tolist(), columns=[f"viome_{i}" for i in range(L)])
            demo = pd.concat([demo.drop(columns=["viome"]), vdf], axis=1)
        demo.drop(columns=[c for c in ["race"] if c in demo], inplace=True, errors="ignore")

        df = cgm.merge(img, on=["subject_id","day"])
        if split=="train":
            df = df.merge(lbl, on=["subject_id","day"])
        df = df.merge(demo, on="subject_id")

        self.tabular_cols = [c for c in demo.columns if c!="subject_id"]
        self.df = df.reset_index(drop=True)
        if split=="train":
            self.scaler = StandardScaler()
            self.df[self.tabular_cols] = self.scaler.fit_transform(self.df[self.tabular_cols])
        else:
            self.scaler = scaler
            if scaler:
                self.df[self.tabular_cols] = scaler.transform(self.df[self.tabular_cols])
        self.split = split

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        cgm_seq = pad_trunc(parse_cgm(r["cgm_data"]))
        cgm_t = torch.tensor(cgm_seq, dtype=torch.float32).unsqueeze(0)  # 1×100
        img_t = parse_img(r["image_before_lunch"])                     # 3×64×64
        tab_t = torch.tensor(r[self.tabular_cols].values.astype(np.float32))
        if self.split=="train":
            y = torch.tensor(r["calories_total"], dtype=torch.float32)
            return cgm_t, img_t, tab_t, y
        else:
            return cgm_t, img_t, tab_t

# 4. MODEL
class CGMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,5,padding=2)
        self.conv2 = nn.Conv1d(16,32,5,padding=2)
        self.pool  = nn.MaxPool1d(2)
        # after two pools: 100→50→25, channels=32
        self.fc    = nn.Linear(25*32, 32)

    def forward(self,x):
        x = F.relu(self.pool(self.conv1(x)))  # →16×50
        x = F.relu(self.pool(self.conv2(x)))  # →32×25
        x = x.flatten(1)
        return F.relu(self.fc(x))             # →32

class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool  = nn.MaxPool2d(2)
        dim = (IMG_SIZE//8)*(IMG_SIZE//8)*64
        self.fc    = nn.Linear(dim, 64)

    def forward(self,x):
        x = F.relu(self.pool(self.conv1(x)))  # →16×32×32
        x = F.relu(self.pool(self.conv2(x)))  # →32×16×16
        x = F.relu(self.pool(self.conv3(x)))  # →64×8×8
        x = x.flatten(1)
        return F.relu(self.fc(x))             # →64

class TabEncoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, 64)
        self.fc2 = nn.Linear(64, 32)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))            # →32

class MultiModalNet(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.cgm_enc = CGMEncoder()
        self.img_enc = ImgEncoder()
        self.tab_enc = TabEncoder(tab_dim)
        self.head    = nn.Sequential(
            nn.Linear(32+64+32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1)
        )

    def forward(self, cgm, img, tab):
        e1 = self.cgm_enc(cgm)
        e2 = self.img_enc(img)
        e3 = self.tab_enc(tab)
        f  = torch.cat([e1,e2,e3], dim=1)
        return self.head(f).squeeze(1)

# 5. TRAIN & SUBMIT
def train_and_evaluate():
    train_ds = MultiModalDataset("train")
    val_df, _ = train_test_split(train_ds.df, test_size=0.2, random_state=42)
    val_ds = MultiModalDataset("train", scaler=train_ds.scaler)
    val_ds.df = val_df

    test_ds = MultiModalDataset("test", scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MultiModalNet(len(train_ds.tabular_cols)).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(1, EPOCHS+1):
        # train
        model.train()
        total=0
        for cgm,img,tab,y in train_loader:
            cgm,img,tab,y = cgm.to(DEVICE), img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(cgm,img,tab)
            loss = loss_fn(pred,y)
            loss.backward()
            opt.step()
            total += loss.item()*cgm.size(0)
        train_rmse = np.sqrt(total/len(train_ds))

        # val
        model.eval()
        total=0
        with torch.no_grad():
            for cgm,img,tab,y in val_loader:
                cgm,img,tab,y = cgm.to(DEVICE), img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
                total += loss_fn(model(cgm,img,tab), y).item()*cgm.size(0)
        val_rmse = np.sqrt(total/len(val_ds))

        print(f"Epoch {ep}/{EPOCHS} — Train RMSE: {train_rmse:.1f}, Val RMSE: {val_rmse:.1f}")

    # predict test
    model.eval()
    preds=[]
    with torch.no_grad():
        for cgm,img,tab in test_loader:
            cgm,img,tab = cgm.to(DEVICE), img.to(DEVICE), tab.to(DEVICE)
            preds.append(model(cgm,img,tab).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    pd.DataFrame({"row_id":np.arange(len(preds)), "label":preds}).to_csv("submission.csv", index=False)
    print("submission.csv saved.")

if __name__=="__main__":
    train_and_evaluate()
