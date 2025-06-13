import ast, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms

# --- 1. CONFIGURATION ---
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR  = "sp-25-tamu-csce-633-600-machine-learning"
CGM_LEN   = 100
IMG_SIZE  = 64
BATCH     = 64
MAX_EPOCH = 100
LR        = 1e-4
PATIENCE  = 20

# --- 2. UTILITIES ---
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def parse_cgm(s):
    try:
        arr = ast.literal_eval(s)
        return [float(v) for _, v in arr]
    except:
        return []

def pad_trunc(seq, L=CGM_LEN):
    return seq[:L] + [0.0]*(L-len(seq)) if len(seq)<L else seq[:L]

def parse_img(s, tfms):
    try:
        img = np.array(ast.literal_eval(s), dtype=np.float32)  # H×W×3
        t = torch.tensor(img).permute(2,0,1)                   # 3×H×W
        return tfms(t)                                        # 3×IMG×IMG
    except:
        return tfms(torch.zeros(3,IMG_SIZE,IMG_SIZE))

# --- 3. DATASET ---
class MultiModalDataset(Dataset):
    def __init__(self, split="train", scaler=None):
        # load CSVs
        cgm_df  = pd.read_csv(os.path.join(DATA_DIR,f"cgm_{split}.csv"), encoding="utf-8-sig")
        demo_df = pd.read_csv(os.path.join(DATA_DIR,f"demo_viome_{split}.csv"), encoding="utf-8-sig")
        img_df  = pd.read_csv(os.path.join(DATA_DIR,f"img_{split}.csv"), encoding="utf-8-sig")
        normalize_cols(cgm_df); normalize_cols(demo_df); normalize_cols(img_df)
        if split=="train":
            lbl_df = pd.read_csv(os.path.join(DATA_DIR,"label_train.csv"), encoding="utf-8-sig")
            normalize_cols(lbl_df)

        # expand viome
        if "viome" in demo_df:
            v = demo_df["viome"].apply(lambda s: [float(x) for x in s.split(",")] if isinstance(s,str) else [])
            L = v.apply(len).max()
            vdf = pd.DataFrame(v.tolist(), columns=[f"viome_{i}" for i in range(L)])
            demo_df = pd.concat([demo_df.drop(columns="viome"), vdf], axis=1)
        demo_df.drop(columns=[c for c in ["race"] if c in demo_df], inplace=True)

        # merge
        df = cgm_df.merge(img_df, on=["subject_id","day"])
        if split=="train":
            df = df.merge(lbl_df, on=["subject_id","day"])
        df = df.merge(demo_df, on="subject_id")

        self.tab_cols = [c for c in demo_df.columns if c!="subject_id"]
        self.df       = df.reset_index(drop=True)

        # scale tabular
        if split=="train":
            self.scaler = StandardScaler()
            self.df[self.tab_cols] = self.scaler.fit_transform(self.df[self.tab_cols])
        else:
            self.scaler = scaler
            if scaler:
                self.df[self.tab_cols] = scaler.transform(self.df[self.tab_cols])

        # image transforms
        if split=="train":
            self.img_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.Resize((IMG_SIZE,IMG_SIZE)),
                transforms.ToTensor()
            ])
        else:
            self.img_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE,IMG_SIZE)),
                transforms.ToTensor()
            ])

        self.split = split

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # CGM
        seq  = pad_trunc(parse_cgm(r["cgm_data"]))
        cgm  = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)  # L×1

        # Image (before lunch)
        img  = parse_img(r["image_before_lunch"], self.img_tfms)     # 3×IMG×IMG

        # tabular
        tab  = torch.tensor(r[self.tab_cols].values.astype(np.float32))

        if self.split=="train":
            y = torch.tensor(r["calories_total"], dtype=torch.float32)
            return cgm, img, tab, y
        return cgm, img, tab

# --- 4. MODEL ---
class CGMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(32*2, 32)
    def forward(self,x):
        # x: B×L×1
        _, (h,_)= self.lstm(x)
        # h: (num_layers*2)×B×32 → take last layer’s both directions
        h_fwd = h[-2]
        h_bwd = h[-1]
        hcat  = torch.cat([h_fwd,h_bwd], dim=1)  # B×64
        return F.relu(self.fc(hcat))             # B×32

class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        dim = (IMG_SIZE//8)*(IMG_SIZE//8)*64
        self.fc = nn.Linear(dim, 64)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return F.relu(self.fc(x))  # B×64

class TabEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
    def forward(self,x): return self.net(x)  # B×32

class MultiModalNet(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.cgm_enc = CGMEncoder()
        self.img_enc = ImgEncoder()
        self.tab_enc = TabEncoder(tab_dim)
        self.head    = nn.Sequential(
            nn.Linear(32+64+32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, cgm, img, tab):
        e1 = self.cgm_enc(cgm)
        e2 = self.img_enc(img)
        e3 = self.tab_enc(tab)
        f  = torch.cat([e1,e2,e3], dim=1)
        return self.head(f).squeeze(1)

# --- 5. TRAIN & EVALUATE ---
def train_and_evaluate():
    # prepare datasets
    train_ds = MultiModalDataset("train")
    val_df, _ = train_test_split(train_ds.df, test_size=0.2, random_state=42)
    val_ds = MultiModalDataset("train", scaler=train_ds.scaler)
    val_ds.df = val_df
    test_ds = MultiModalDataset("test", scaler=train_ds.scaler)

    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=BATCH)
    te = DataLoader(test_ds,  batch_size=BATCH)

    model = MultiModalNet(len(train_ds.tab_cols)).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCH)
    loss_fn = nn.MSELoss()

    best_rmse, wait = float("inf"), 0
    for ep in range(1, MAX_EPOCH+1):
        # train
        model.train()
        total=0
        for cgm,img,tab,y in tl:
            cgm,img,tab,y = cgm.to(DEVICE), img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(cgm,img,tab), y)
            loss.backward()
            opt.step()
            total += loss.item()*cgm.size(0)
        train_rmse = np.sqrt(total/len(train_ds))

        # val
        model.eval()
        total=0
        with torch.no_grad():
            for cgm,img,tab,y in vl:
                cgm,img,tab,y = cgm.to(DEVICE), img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
                total += loss_fn(model(cgm,img,tab), y).item()*cgm.size(0)
        val_rmse = np.sqrt(total/len(val_ds))

        sched.step()
        print(f"Epoch {ep}/{MAX_EPOCH} – train {train_rmse:.1f}, val {val_rmse:.1f}")

        # early stopping
        if val_rmse < best_rmse:
            best_rmse, wait = val_rmse, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # load best and predict
    model.load_state_dict(torch.load("best.pt"))
    model.eval()
    preds=[]
    with torch.no_grad():
        for cgm,img,tab in te:
            cgm,img,tab = cgm.to(DEVICE), img.to(DEVICE), tab.to(DEVICE)
            preds.append(model(cgm,img,tab).cpu().numpy())
    preds = np.concatenate(preds,axis=0)
    pd.DataFrame({"row_id":np.arange(len(preds)),"label":preds})\
      .to_csv("submission.csv", index=False)
    print("submission.csv saved.")

if __name__=="__main__":
    train_and_evaluate()
