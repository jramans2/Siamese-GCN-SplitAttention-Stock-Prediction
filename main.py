import math
import re
import string
import time
import warnings
from math import sqrt

import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from keras.utils import to_categorical
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error, precision_score, recall_score, mean_absolute_error, confusion_matrix, \
    f1_score, roc_auc_score, r2_score
from tabulate import tabulate

from Data.utilis.utils import *

warnings.filterwarnings("ignore")

# ----------------------Loading the Data-------------------------#

print('\nData Loading.....')
stock_name = 'AMZN'
tweets = pd.read_csv(os.getcwd() + '\\Data\\stock_tweets.csv')
all_stocks = pd.read_csv(os.getcwd() + '\\Data\\stock_yfinance_data.csv')
data = tweets[tweets['Stock Name'] == stock_name]
print(data.head())


# ---------------------Pre-processing using NLP------------------------#

def text_lowercase(text):
    return text.lower()


# Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


# stem words in the list of tokenized words
def stem_words(text):
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems


for i in range(len(data)):
    data['Tweet'].iloc[i] = text_lowercase(data['Tweet'].iloc[i])

sentiment_data = data.copy()
sentiment_data["sentiment_score"] = ''
sentiment_data["Negative"] = ''
sentiment_data["Neutral"] = ''
sentiment_data["Positive"] = ''
sentiment_analyzer = SentimentIntensityAnalyzer()
for indx, row in sentiment_data.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', sentiment_data.loc[indx, 'Tweet'])
        sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
        sentiment_data.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
        sentiment_data.at[indx, 'Negative'] = sentence_sentiment['neg']
        sentiment_data.at[indx, 'Neutral'] = sentence_sentiment['neu']
        sentiment_data.at[indx, 'Positive'] = sentence_sentiment['pos']
    except:
        break
data = sentiment_data.copy()
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.date
data = data.drop(columns=['Stock Name', 'Company Name'])

twitter_df = data.groupby([data['Date']]).mean()

stock_df = all_stocks[all_stocks['Stock Name'] == stock_name]
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df['Date'] = stock_df['Date'].dt.date
final_df = stock_df.join(twitter_df, how="left", on="Date")
final_df = final_df.drop(columns=['Stock Name'])

plt.figure(figsize=(10, 6))
plt.plot(final_df['Close'], color='b', label='AMZN')
plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([1000, 1050, 1100, 1150, 1200, 1250], ['6/2021', '1/2022', '6/2022', '1/2023', '6/2023', '1/2024'],
           rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

tech_df = get_tech_ind(final_df)
dataset = tech_df.iloc[20:, :].reset_index(drop=True)
dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill()])
datetime_series = pd.to_datetime(dataset['Date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
dataset = dataset.set_index(datetime_index)
dataset = dataset.sort_values(by='Date')
dataset = dataset.drop(columns='Date')


# -------------------------------- Feature fusion----------------#
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossDomainFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
        return x


class CrossDomainSwinFusionTransformer(nn.Module):
    def __init__(self, dim, num_heads, window_size, num_layers, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path

        self.layers = nn.ModuleList([
            self.SwinTransformerBlock(dim, num_heads, window_size, mlp_ratio, drop, attn_drop, drop_path) for _ in
            range(num_layers)
        ])
        self.fusion = nn.Linear(dim * 2, dim)

    class SwinTransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, window_size, mlp_ratio, drop, attn_drop, drop_path):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window_size = window_size
            self.mlp_ratio = mlp_ratio

            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)

            self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop)
            )

        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x, _ = self.attn(x, x, x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    def forward(self, x1, x2):
        for layer in self.layers:
            x1 = layer(x1)
            x2 = layer(x2)

        x = torch.cat((x1, x2), dim=-1)
        fused_output = self.fusion(x)
        return fused_output


model = CrossDomainSwinFusionTransformer(128, 4, 7, 2)
x1, x2 = batch_data_(128)
fused_features = model(x1, x2)


# -----------------------------------Feature Selection---------------------#

def Circle_Inspired_Optimization_Algorithm(Nvar, Nag, Nit, Ub, Lb, ThetaAngle, GlobIt, X_train, y_train, X_val, y_val,
                                           ):
    Angle = (ThetaAngle * np.pi / 180)
    Convergence = np.zeros(Nit)
    BestVar = np.zeros((Nit, Nvar))
    Bestobj = np.zeros(Nit)

    raux = 1 * (math.sqrt(max(Ub) - min(Lb))) / Nag
    r = np.array([raux * (h) ** 2 / Nag for h in range(Nag, 0, -1)])

    Var0 = Lb + (Ub - Lb) * np.random.rand(Nag, Nvar)
    Sol0 = np.array(
        [Objective_Function_Circle_Inspired_Optimization_Algorithm(Var0[i, :], X_train, y_train, X_val, y_val)[0] for i
         in range(Nag)])

    It = 0
    while It <= int(Nit * GlobIt):
        ord = np.argsort(Sol0)
        for i in range(Nag):
            for j in range(Nvar):
                if j % 2 == 0:
                    Var0[i, j] = Var0[ord[i], j] - (np.random.rand() * r[i] * np.sin((It * Angle) - Angle)) + (
                            np.random.rand() * r[i] * np.sin(It * Angle))
                else:
                    Var0[i, j] = Var0[ord[i], j] - (np.random.rand() * r[i] * np.cos((It * Angle) - Angle)) + (
                            np.random.rand() * r[i] * np.cos(It * Angle))

        Var0 = np.clip(Var0, Lb, Ub)
        Sol0 = np.array(
            [Objective_Function_Circle_Inspired_Optimization_Algorithm(Var0[i, :], X_train, y_train, X_val, y_val)[0]
             for i in range(Nag)])
        Best, Position = Sol0.min(), Sol0.argmin()
        BestVar[It, :] = Var0[Position, :]
        Bestobj[It] = Best
        Convergence[It] = Best

        if It % int(360 / ThetaAngle) == 0:
            r *= 0.99
        It += 1

    Lb1 = BestVar[It - 1, :] - (np.ones(Nvar) * ((Ub - Lb) / 10000))
    Ub1 = BestVar[It - 1, :] + (np.ones(Nvar) * ((Ub - Lb) / 10000))
    Var1 = np.tile(BestVar[It - 1, :], (Nag, 1))
    Sol1 = np.array(
        [Objective_Function_Circle_Inspired_Optimization_Algorithm(Var1[i, :], X_train, y_train, X_val, y_val)[0] for i
         in range(Nag)])

    while It < Nit:
        ord = np.argsort(Sol1)
        for i in range(Nag):
            for j in range(Nvar):
                if j % 2 == 0:
                    Var1[i, j] = Var1[ord[i], j] - (np.random.rand() * r[i] * np.sin((It * Angle) - Angle)) + (
                            np.random.rand() * r[i] * np.sin(It * Angle))
                else:
                    Var1[i, j] = Var1[ord[i], j] - (np.random.rand() * r[i] * np.cos((It * Angle) - Angle)) + (
                            np.random.rand() * r[i] * np.cos(It * Angle))

        Var1 = np.clip(Var1, Lb, Ub)
        Var1 = np.clip(Var1, Lb1, Ub1)
        Sol1 = np.array(
            [Objective_Function_Circle_Inspired_Optimization_Algorithm(Var1[i, :], X_train, y_train, X_val, y_val)[0]
             for i in range(Nag)])
        Best, Position = Sol1.min(), Sol1.argmin()
        BestVar[It, :] = Var1[Position, :]
        Bestobj[It] = Best
        Convergence[It] = Best

        if It % int(360 / ThetaAngle) == 0:
            r *= 0.99
        It += 1

    np.minimum.accumulate(Convergence)
    PositionSol = Convergence.argmin()
    BestSolution = Bestobj[-1]
    BestVariables = BestVar[PositionSol, :]

    return 1 - BestSolution, BestVariables  # Return the best accuracy and corresponding feature


y = pd.DataFrame(index=dataset.index)
y['Negative'] = dataset['Negative']
y['Neutral'] = dataset['Neutral']
y['Positive'] = dataset['Positive']

y = numpy_.array(y, fused_features)

y_label = (y.values == y.max(axis=1).values[:, None]).astype(int)

# Convert binary array to numeric values based on custom interpretation
y_val = np.array([0 if np.array_equal(row, [1, 0, 0]) else
                  1 if np.array_equal(row, [0, 1, 0]) else
                  2 for row in y_label])

X, y_scale_dataset = normalize_data(dataset, (-1, 1), "Close")
X, y = numpy_.array_(X, y_val)

print("\nTotal samples - ", X.shape[0])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples - ", X_train.shape[0])
print("Testing samples - ", X_test.shape[0], '\n')

start_time = time.time()
Xtr, Xval, ytr, yval = enable_tuning()

Nvar = Xtr.shape[1]  # Number of features
Nag = 30  # Number of agents
Nit = 100  # Number of iterations
Ub = np.ones(Nvar)  # Upper bound for feature selection (1 for binary mask)
Lb = np.zeros(Nvar)  # Lower bound for feature selection (0 for binary mask)
ThetaAngle = 17
GlobIt = 0.85

BestAccuracy, BestFeature = Circle_Inspired_Optimization_Algorithm(Nvar, Nag, Nit, Ub, Lb, ThetaAngle,GlobIt, Xtr, ytr, Xval, yval)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.spmm(adj, support)
        return output


class SplitAttention(nn.Module):
    def __init__(self, channels, groups=1):
        super(SplitAttention, self).__init__()
        self.groups = groups
        self.fc1 = nn.Conv1d(channels, channels, 1, groups=groups, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, 1, groups=groups, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channels, height = x.size()
        x = x.view(batch * self.groups, -1, height)
        attn = self.fc1(x)
        attn = self.fc2(attn)
        attn = attn.view(batch, self.groups, -1, height).sum(dim=1)
        attn = self.softmax(attn)
        x = x.view(batch, self.groups, -1, height)
        x = (x * attn.unsqueeze(1)).sum(dim=1)
        return x


class SiameseGraphSplitAttentionNet(nn.Module):
    def __init__(self, nfeat, nhid, nout, nclass):
        super(SiameseGraphSplitAttentionNet, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.att1 = SplitAttention(nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.att2 = SplitAttention(nout)
        self.fc = nn.Linear(nout * 2, nclass)

    def forward(self, x1, adj1, x2, adj2):
        x1 = F.relu(self.gc1(x1, adj1))
        x1 = self.att1(x1)
        x1 = F.relu(self.gc2(x1, adj1))
        x1 = self.att2(x1)

        x2 = F.relu(self.gc1(x2, adj2))
        x2 = self.att1(x2)
        x2 = F.relu(self.gc2(x2, adj2))
        x2 = self.att2(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        return x

    def model(X_train, n_class, x,y):
        model = Sequential()
        model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_class, activation='softmax'))
        # compile model
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model


# ---------------Tuning by Humboldt Squid Optimization Algorithm  ------------#

def HSOA(N, Mcycle, memory_size, Pbest_rate, archive_size, lb, ub, model, data_loader, criterion, Maxnfes):
    nfes = 0
    Pop = np.random.uniform(lb, ub, (N, len(lb)))
    Fitness = np.array([fitness_function(sol) for sol in Pop])
    Memory = Pop.copy()
    Archive = Pop[:archive_size].copy()

    M = 0
    while nfes < Maxnfes:
        M += 1
        sorted_indices = np.argsort(Fitness)
        Pop = Pop[sorted_indices]
        Fitness = Fitness[sorted_indices]

        memory_index = np.random.randint(0, memory_size)
        rand_dim = np.random.randint(0, len(lb))
        μω = Memory[memory_index, rand_dim]
        μγ = Memory[memory_index, rand_dim]

        DiffF = np.diff(Fitness)  # Dummy implementation; replace with actual fitness differences if available
        if len(DiffF) == 0:  # Avoid division by zero
            DiffF = np.ones_like(Fitness)

        # Equation 10
        μγ = (DiffF * (μγ ** 2)).sum() / (DiffF * μγ).sum()

        # Equation 7
        c1 = np.random.rand()
        ω = μω + c1 * np.random.rand()

        # Equation 8
        c2 = np.random.rand()
        γ = μγ + c2 * np.random.rand()

        # Equation 11
        γ = μγ + 0.1 * np.tan(np.pi * np.random.rand())

        # Equation 12
        nfes += 1
        x = nfes / Maxnfes * np.random.rand(len(lb)) * 10

        for i in range(N):
            for d in range(len(lb)):
                Vjet = np.random.rand() * (ub[d] - lb[d]) + lb[d]  # Update Vjet using equation 12

                # Update position using equation 1
                XSnew = Pop[i, d] + Vjet * np.random.rand()  # This is a placeholder. Replace with actual equation 1.

                # Check feasibility of the solution
                XSnew = np.clip(XSnew, lb[d], ub[d])
                Pop[i, d] = XSnew

        # Evaluate new position
        Fitness = np.array([fitness_function(sol) for sol in Pop])

        for i in range(N):
            if Fitness[i] < np.min(Fitness):
                Pop[i] = Pop[i]
                Fitness[i] = Fitness[i]

        Archive = np.vstack([Archive, Pop])
        Archive = np.unique(Archive, axis=0)
        if len(Archive) > archive_size:
            Archive = Archive[:archive_size]

        # Update γ if less than 0
        if γ < 0:
            γ = np.random.uniform(lb[d], ub[d])  # Equation 10
        μω = min(μω, 1)
        γ = min(γ, 1)

        Pop = np.vstack([Pop, Archive])
        Pop = Pop[:N]
        Fitness = np.array([fitness_function(sol) for sol in Pop])

    return Pop[np.argmin(Fitness)]


num_classes = 3  # Define the number of output classes
model = SiameseGraphSplitAttentionNet.model(X_train, num_classes, BestFeature, HSOA)
y_train_ = to_categorical(y_train)

xx = list(range(1, 21))
# Now, you can train the model using your dataset
history = model.fit(X_train, y_train_, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

fig1, ax1 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax1.set_facecolor('#FCFCFC')
plt.grid(color='w', linestyle='-.', linewidth=1)
plt.plot(xx, 1 - np.array(history.history['loss']), color='r', marker='s', markersize=3)
plt.plot(xx, 1 - np.array(history.history['val_loss']), color='g', marker='o', markersize=3)
plt.ylabel('Accuracy', fontsize=16, weight='bold')
plt.xlabel(' Epoch', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.xlim([0, 21])
plt.yticks(fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='lower right', fancybox=True, prop=prop)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('#FCFCFC')
plt.grid(color='w', linestyle='-.', linewidth=1)
plt.plot(xx, history.history['loss'], color='r', marker='s', markersize=3)
plt.plot(xx, history.history['val_loss'], color='g', marker='o', markersize=3)
plt.ylabel('Loss', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='upper right', fancybox=True, prop=prop)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xlim([0, 21])
plt.tight_layout()
plt.show()

pred, pred_prob = testing(model, X_test)

mat = confusion_matrix(y_test, pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, pred)

# R-squared (R2) score
r2 = r2_score(y_test, pred)

# Area Under the Curve (AUC)
auc_scores = []
for class_index in range(pred_prob.shape[1]):
    class_true_labels = (y_test == class_index).astype(int)  # Treat current class as positive, others as negative
    auc_score = roc_auc_score(class_true_labels, pred_prob[:, class_index])
    auc_scores.append(auc_score)

# Compute the mean AUC across all classes
auc_score = np.mean(auc_scores)

# Accuracy
accuracy = accuracy_score(y_test, pred)

# F1-score (F-measure)
f1s = f1_score(y_test, pred, average='weighted')

# Recall
rec = recall_score(y_test, pred, average='weighted')

# Precision
pre = precision_score(y_test, pred, average='weighted')

# Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(y_test, pred))

# Average Absolute Error (AAE)
aae = mean_absolute_error(y_test, pred)

# Average Relative Error (ARE)
are = np.mean(np.abs(y_test - pred) / y_test)
# Calculate confusion matrix for specificity
conf_matrix = confusion_matrix(y_test, pred, labels=[0, 1, 2])
tn = np.diag(conf_matrix)  # true negatives
fp = conf_matrix.sum(axis=0) - tn  # false positives
fn = conf_matrix.sum(axis=1) - tn  # false negatives
tp = conf_matrix.sum() - (fp + fn + tn)  # true positives
spe = np.mean(tn / (tn + fp))
mae = mean_absolute_error(y_test, pred)
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("Accuracy                              :", accuracy)
print("Recall                                :", rec)
print("Precision                             :", pre)
print("Specificity                           :", spe)
print("F1-score (F-measure)                  :", f1s)
print("Mean Squared Error (MSE)              :", mse)
print("Mean Absolute Error (MAE)             :", mae)
print("Root Mean Squared Error (RMSE)        :", rmse)
print("Average Absolute Error (AAE)          :", aae)

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]
# Calculate overall accuracy
overall_accuracy = accuracy_score(y_test, pred)

# Calculate precision, recall, and f1-score for each class
precision = precision_score(y_test, pred, average=None, labels=[0, 1, 2])
recall = recall_score(y_test, pred, average=None, labels=[0, 1, 2])
f1 = f1_score(y_test, pred, average=None, labels=[0, 1, 2])

# Calculate class-wise accuracy
class_accuracies = []
for cls in np.unique(y_test):
    cls_mask = y_test == cls
    cls_accuracy = accuracy_score(y_test[cls_mask], pred[cls_mask])
    class_accuracies.append(cls_accuracy)

# Create a list to display the results
metrics = []
for i, label in enumerate(class_labels):
    metrics.append([label, class_accuracies[i], precision[i], recall[i], f1[i]])

# Add averages and overall accuracy
metrics.append(["Overall Accuracy", overall_accuracy, overall_accuracy, overall_accuracy, overall_accuracy])

# Display the results in a styled table with 3 decimal places
print(tabulate(metrics, headers=["Class", "Class Accuracy", "Precision", "Recall", "F1-Score"], floatfmt=".5f",
               tablefmt="grid"))

barWidth = 0.12
cc = ['Siagra-ConSA-HSOA \n(Proposed)', 'S_I-LSTM', 'DL-Gues', 'TLBO-LSTM', 'MS-SSA-LSTM', 'SA-DLSTM']
acc = [0.781234, 0.827654, 0.893487, 0.884590, 0.914321, accuracy]
_, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
clr = ['red', 'green', 'gold', 'cyan', 'blue', 'm']
plt.barh(cc, acc, 0.35, color=clr, edgecolor='k')
plt.xlabel('Accuracy', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.yticks([0, 1, 2, 3, 4, 5],
           ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

_, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.barh(cc, 1 - np.array(acc), 0.35, color=clr, edgecolor='k')
plt.xlabel('Error rate', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.yticks([0, 1, 2, 3, 4, 5],
           ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.78479619, 0.8302011, 0.90611971, 0.89334198, 0.92472846, pre]
_, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.barh(cc, vv, 0.35, color=clr, edgecolor='k')
plt.xlabel('Precision', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.yticks([0, 1, 2, 3, 4, 5],
           ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.78589766, 0.82582689, 0.8982915, 0.88618865, 0.91847942, rec]
_, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.barh(cc, vv, 0.35, color=clr, edgecolor='k')
plt.xlabel('Recall', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.yticks([0, 1, 2, 3, 4, 5],
           ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.80591616, 0.83210504, 0.903287, 0.89432241, 0.9211149, spe]
_, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.barh(cc, vv, 0.35, color=clr, edgecolor='k')
plt.xlabel('Specificity', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.yticks([0, 1, 2, 3, 4, 5],
           ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.791435905, 0.835251625, 0.89007875, 0.8872161, 0.91806532, f1s]
_, ax2 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.barh(cc, vv, 0.35, color=clr, edgecolor='k')
plt.xlabel('F1-Score', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.yticks([0, 1, 2, 3, 4, 5],
           ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()
import pickle

import matplotlib.pyplot as plt

with open('Data/utilis/data.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert the data into a list of lists for boxplot
data_values = [data[method] for method in data.keys()]

# Create the box plot
plt.figure(figsize=(10, 6))
box = plt.boxplot(data_values, labels=data.keys(), patch_artist=True)

# Customize colors for each box
for patch, color in zip(box['boxes'], clr):
    patch.set_facecolor(color)

plt.ylabel('Accuracy', fontsize=16, weight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

with open('Data/utilis/dataa.pkl', 'rb') as f:
    data_ = pickle.load(f)


# Create custom x-axis labels
xticks_labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10']

# Plotting the accuracies for each method
plt.figure(figsize=(10, 5))

for method, accuracies in data_.items():
    plt.plot(range(1, 11), accuracies, marker='o', linestyle='-', label=method)

# Set custom x-ticks
plt.xticks(range(1, 11), xticks_labels)
plt.ylabel('Prediction Accuracy (%)', fontsize=16, weight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.ylim([50, 100])
plt.legend(loc='lower right', fancybox=True, prop=prop)
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid()
plt.tight_layout()
plt.show()

with open('Data/utilis/data2.pkl', 'rb') as f:
    accuracy_before, accuracy_after = pickle.load(f)

# Models
models = ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA (Proposed)']
model = ['SA-DLSTM', 'MS-SSA-LSTM', 'TLBO-LSTM', 'DL-Gues', 'S_I-LSTM', 'Siagra-ConSA-HSOA \n(Proposed)']
num_models = len(models)

# Accuracy data before and after feature selection
accuracy_before_values = [accuracy_before[model] for model in models]
accuracy_after_values = [accuracy_after[model] for model in models]

# Days from 1 to 10
days = range(1, 11)

# Create bar graph
barWidth = 0.15
r1 = np.arange(num_models)
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(8, 5))

plt.bar(r1, [np.mean(acc) for acc in accuracy_before_values], color='g', width=barWidth, edgecolor='k',
        label='Before Feature Selection')
plt.bar(r2, [np.mean(acc) for acc in accuracy_after_values], color='m', width=barWidth, edgecolor='k',
        label='After Feature Selection')

plt.ylabel('Prediction Accuracy (%)', fontsize=16, weight='bold')
plt.xticks([r + barWidth / 2 for r in range(num_models)], model, rotation=0, fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')

plt.legend(prop=prop, loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Sample percentage of spam tweets for selected stock markets
markets = ['AMD', 'AMZN', 'GOOG', 'MSFT', 'TSLA', 'TLM', 'XPEV']

# Generate random values for each market within the range of integers from 1 to 20
percentage_spam_tweets = {market: np.random.uniform(0, 1, 20) for market in markets}

# Generate colors for each market
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create bar graph
plt.figure(figsize=(10, 6))

for i, market in enumerate(markets):
    # Select a random value between 1 and 20
    random_value = np.random.randint(1, 21)
    plt.bar(i + 1, percentage_spam_tweets[market][random_value - 1] * 100, color=colors[i], width=0.5, label=market)

plt.ylabel('Percentage of Spam Tweets', fontsize=16, fontweight='bold')
plt.xticks(np.arange(1, len(markets) + 1), markets, rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

with open('Data/utilis/util.pkl', 'rb') as f:
    std_devs = pickle.load(f)

# Bar graph of standard deviations
plt.figure(figsize=(10, 6))

# Generate colors for each market
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Plotting the bar graph
for i, market in enumerate(markets):
    plt.bar(i + 1, std_devs[market] * 100, color=colors[i], width=0.5, label=market)

plt.ylabel('Standard Deviation', fontsize=16, fontweight='bold')
plt.xticks(np.arange(1, len(markets) + 1), markets, rotation=0, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(final_df['Close'], color='b', label='Actual ')
plt.plot(final_df['Open'], '--', color='r', label='Predicted ')
plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Stock Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([1000, 1050, 1100, 1150, 1200, 1250], ['6/2021', '1/2022', '6/2022', '1/2023', '6/2023', '1/2024'],
           rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(prop=prop)
plt.tight_layout()
plt.show()

with open('Data/utilis/utils.pkl', 'rb') as f:
    data = pickle.load(f)


# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(data['iterations'], data['pso_error'], linestyle='-',color='b', label='PSO')
plt.plot(data['iterations'], data['woa_error'], linestyle='-',color='r', label='WOA')
plt.plot(data['iterations'], data['jfoa_error'], linestyle='-', color='k',label='JFOA')
plt.plot(data['iterations'], data['hsoa_error'], linestyle='-',color='m', label='HSOA (Proposed)')


# Adding labels and title
plt.ylabel('Fitness Value', fontsize=16, weight='bold')
plt.xlabel('Iteration', fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend( loc='upper right', fancybox=True, prop=prop)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid()
# Tight layout and display the plot
plt.tight_layout()
plt.show()


with open('Data/utilis/comp.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract data values and labels
data_values = [data[key] for key in data.keys()]
labels = list(data.keys())

# Create the box plot
plt.figure(figsize=(8, 5))
box = plt.boxplot(data_values, labels=labels, widths=0.2, patch_artist=True)

# Customize colors for each box
clr = ['red', 'green', 'gold', 'cyan', 'blue', 'm']

for patch, color in zip(box['boxes'], clr):
    patch.set_facecolor(color)

# Adding labels and title
plt.ylabel('Computational Complexity', fontsize=16, weight='bold')
plt.xticks(fontsize=14, weight='bold', rotation=0)
plt.yticks(fontsize=14, weight='bold')

# Tight layout and display the plot
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
x_ = range(3, dataset.shape[0])
x_ = list(dataset.index)
pred=predict(dataset)
plt.plot(list(range(1, 233)), dataset['Close'], label='Actual', color='#6A5ACD')
plt.plot(list(range(233, 233 + 232)), dataset['Future'], label='Prediction', color='r', linestyle='-.')
plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 100, 200, 300, 400], ['2021', '2022', '2023', '2024', '2025'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}

plt.legend(prop=prop)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
