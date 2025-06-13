import ast, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Device set to use CUDA if available otherwise CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Path to the folder containing all CSV data
DATA_DIR = "dataFiles"
# Fixed length for CGM time data (Typical length ~96, set to 100 to keep it consistent)
CGM_LEN = 100
# Image resize
IMG_SIZE = 64
# No of samples per batch
BATCH = 64
# Maximum number of epochs 
MAX_EPOCH = 100
# Initial learning rate for optimizer
LR = 1e-4
# Epocjs to wait for early stopping
PATIENCE = 20

# Fn to clean df column name
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

#Fn to parse CGM data into a list and remove timestamp as the intervals are fixed ~5mins
def parse_cgm(s):
    try:
        arr = ast.literal_eval(s)
        return [float(v) for k, v in arr]
    except:
        return []

# Fn to match the fixed length of CGM data (if short: add 0s else truncate)
def match(seq, L = CGM_LEN):
    return seq[:L] + [0.0]*(L-len(seq)) if len(seq) < L else seq[:L]

# Fn to parse image data into a tensor
def parse_img(s, tfms):
    try:
        img = np.array(ast.literal_eval(s), dtype = np.float32) # HxWx3 array
        t = torch.tensor(img).permute(2,0,1) # to 3xHxW
        return tfms(t)
    except:
        # On failure, return a blank image tensor
        return tfms(torch.zeros(3, IMG_SIZE, IMG_SIZE))

# Class to create a generic dataset for the 3 modalities
class MultiModalDataset(Dataset):
    def __init__(self, split = "train", scaler = None):
        # Read CSVs for each modality
        cgm_df  = pd.read_csv(os.path.join(DATA_DIR,f"cgm_{split}.csv"), encoding = "utf-8-sig")
        normalize_cols(cgm_df) 
        demo_df = pd.read_csv(os.path.join(DATA_DIR,f"demo_viome_{split}.csv"), encoding = "utf-8-sig")
        normalize_cols(demo_df)
        img_df  = pd.read_csv(os.path.join(DATA_DIR,f"img_{split}.csv"), encoding = "utf-8-sig")
        normalize_cols(img_df)

        # If training, then also read the labels
        if split == "train":
            lbl_df = pd.read_csv(os.path.join(DATA_DIR,"label_train.csv"), encoding = "utf-8-sig")
            normalize_cols(lbl_df)

        # Expand the "viome" column into separate numeric columns
        if "viome" in demo_df:
            v = demo_df["viome"].apply(lambda s: [float(x) for x in s.split(",")] if isinstance(s,str) else [])
            l = v.apply(len).max()
            vdf = pd.DataFrame(v.tolist(), columns = [f"viome_{i}" for i in range(l)])
            demo_df = pd.concat([demo_df.drop(columns = "viome"), vdf], axis = 1)

        # Dropping Race column (highly imabalanced and not useful in my opinion)
        demo_df.drop(columns = [c for c in ["race"] if c in demo_df], inplace = True)

        # Merge the three modalities into one DataFrame based on subject_id and day
        df = cgm_df.merge(img_df, on = ["subject_id","day"])
        if split == "train":
            df = df.merge(lbl_df, on = ["subject_id","day"])
        df = df.merge(demo_df, on = "subject_id")

        # Separating demo data
        self.demo_cols = [c for c in demo_df.columns if c != "subject_id"]
        self.df = df.reset_index(drop = True)

        # Scaling demo data in training and reusing the same scaler for test data
        if split == "train":
            self.scaler = StandardScaler()
            self.df[self.demo_cols] = self.scaler.fit_transform(self.df[self.demo_cols])
        else:
            self.scaler = scaler
            if scaler:
                self.df[self.demo_cols] = scaler.transform(self.df[self.demo_cols])

        # Defining image transforms for train and test
        if split == "train":
            self.img_tfms = transforms.Compose([
                transforms.ToPILImage(),               
                transforms.RandomHorizontalFlip(),    
                transforms.RandomRotation(20), # Worked well in my CNN assignment, not sure why it works wonders.  
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

    # Total number of samples
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        # CGM modality
        seq = match(parse_cgm(r["cgm_data"]))
        cgm = torch.tensor(seq, dtype = torch.float32)
        cgm = cgm.unsqueeze(-1) # For LSTM

        # Image modality (before lunch)
        img = parse_img(r["image_before_lunch"], self.img_tfms)

        # Demo modality
        demo = torch.tensor(r[self.demo_cols].values.astype(np.float32))

        # If training, then also return the target calories
        if self.split == "train":
            y = torch.tensor(r["calories_total"], dtype = torch.float32)
            return cgm, img, demo, y

        # Else for testing, no label
        return cgm, img, demo

# Class to create a 1D CGM encoder projecting to 32dim vector using LSTM
class CGMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 32,num_layers = 2, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(32*2, 32)

    def forward(self, x):
        _, (hid, _) = self.lstm(x)
        hid_fwd = hid[-2]
        hid_bwd = hid[-1] 
        hidcat  = torch.cat([hid_fwd, hid_bwd], dim = 1)  # batch_size x 64
        return F.relu(self.fc(hidcat)) # batch_size x 32

# Class to create a 3block CNN encoder projecting to 64dim vector
class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding = 1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding = 1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        
        #Calculate final dims after 3 blocks of conv layers and maxpooling
        dim = (IMG_SIZE//8) * (IMG_SIZE//8) * 64
        self.fc = nn.Linear(dim, 64)

    def forward(self, x):
        x = self.block1(x)       
        x = self.block2(x)       
        x = self.block3(x)       
        x = x.flatten(1)         
        return F.relu(self.fc(x)) # batch_size x 64

# Class to create a 2leayer MLP encoder projecting to 32dim vector
class DemoEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)  # batch_size x 32

# Class to combine 3 encoders and run through a 3layer MLP to predict calories
class MultiModalNet(nn.Module):
    def __init__(self, demo_dim):
        super().__init__()
        self.cgm_enc = CGMEncoder()
        self.img_enc = ImgEncoder()
        self.demo_enc = DemoEncoder(demo_dim)
        self.head = nn.Sequential(
            nn.Linear(32 + 64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, cgm, img, demo):
        e1 = self.cgm_enc(cgm) # CGM features (32)
        e2 = self.img_enc(img) # Image features (64)
        e3 = self.demo_enc(demo) # Demo features (32)
        f  = torch.cat([e1, e2, e3], dim = 1) # batch_size x 128
        return self.head(f).squeeze(1) # batch

# Main Fn to train and evaluate the model
def main():
    train_ds = MultiModalDataset("train")
    # Split out 20% for validation (random, reproducible)
    val_df, _ = train_test_split(train_ds.df, test_size = 0.2, random_state = 42)
    # Create validation dataset, reusing the same scaler
    val_ds = MultiModalDataset("train", scaler = train_ds.scaler)
    val_ds.df = val_df
    # Create test dataset with no labels
    test_ds = MultiModalDataset("test", scaler = train_ds.scaler)

    # Data loaders for each modality
    tr = DataLoader(train_ds, batch_size = BATCH, shuffle = True)
    vl = DataLoader(val_ds, batch_size = BATCH)
    te = DataLoader(test_ds, batch_size = BATCH)

    # Instantiate model & move to CUDA
    model = MultiModalNet(len(train_ds.demo_cols)).to(DEVICE)
    # Adam optimizer
    opt = optim.Adam(model.parameters(), lr = LR)
    # Cosine learning-rate schedule
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = MAX_EPOCH)
    # Mean squared error loss
    loss_fn = nn.MSELoss()

    best_rmrse, wait = float("inf"), 0

    train_losses, val_losses = [], []
    # Main training loop
    for ep in range(1, MAX_EPOCH+1):
        # TRAIN
        model.train()
        total = 0
        count = 0
        for cgm, img, demo, y in tr:
            cgm, img, demo, y = cgm.to(DEVICE), img.to(DEVICE), demo.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(cgm, img, demo)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += ((pred - y) / y).pow(2).sum().item()
            count += y.size(0)                             
        train_rmrse = np.sqrt(total / count)               
        train_losses.append(train_rmrse)

        # VALIDATION
        model.eval()
        total = 0
        with torch.no_grad():
            for cgm, img, demo, y in vl:
                cgm, img, demo, y = cgm.to(DEVICE), img.to(DEVICE), demo.to(DEVICE), y.to(DEVICE)
                pred = model(cgm, img, demo)
                total += ((pred - y) / y).pow(2).sum().item()
        count += y.size(0)  
        val_rmrse = np.sqrt(total / count)           
        val_losses.append(val_rmrse)
        # Step learning-rate scheduler
        sched.step()
        print(f"Epoch {ep}/{MAX_EPOCH} -- train {train_rmrse:.3f}, val {val_rmrse:.3f}")

        # Early stopping check
        if val_rmrse < best_rmrse:
            best_rmrse, wait = val_rmrse, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping!")
                break

    # After training, load the best model weights
    model.load_state_dict(torch.load("best.pt"))
    model.eval()

    # Plot training and validation loss curves
    plot_loss_curve(train_losses, val_losses)

    # Predict on test set
    preds = []
    with torch.no_grad():
        for cgm, img, demo in te:
            cgm, img, demo = cgm.to(DEVICE), img.to(DEVICE), demo.to(DEVICE)
            preds.append(model(cgm, img, demo).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # CSV for Kaggle submission
    pd.DataFrame({"row_id": np.arange(len(preds)), "label": preds}).to_csv("submission.csv", index=False)
    print("submission.csv saved :)")

# Fn to plot training and validation loss curves
def plot_loss_curve(train_losses, val_losses):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()  
    plt.plot(epochs, train_losses, label="Train RMRSE")
    plt.plot(epochs, val_losses,   label="Val RMRSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMRSE")
    plt.title("Training vs. Validation RMRSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()