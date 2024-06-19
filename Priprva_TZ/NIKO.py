"""Tematika podatkov: Uporaba kreditnih kartic
Upravitelje bank skrbi, ker vse več strank opušča uporabo kreditnih kartic. Zato bi bilo koristno, če bi lahko iz obstoječih podatkov odkrili, kdo
namerava prenehati z uporabo kreditne kartice, da bi mu lahko ponudili boljši paket in ugodnosti, ter ga tako odvrnili od opustitve.
Podatki
V treh datotekah ( clientInfo.txt , clientActivity.xlsx , clientTransactions.csv ) se nahajajo podatki o klientih in njihovih dejavnostih
povezenih z uporabo kreditne kartice. Prva datoteka vsebuje splošne podatke o klientu, druga datoteka vsebuje podatke o klientovih ativnostih
povezanih z banko, tretja datoteka pa vsebuje podatke o klientovih transakcijah.
V pomoč pri razumevanju posameznih spremenljivk so vam lahko naslednji opisi ter vrsta podatkov (številski/kategorični
podatek):
Accaunt_Open - ali ima klient pri banki še odprt račun ('yes', 'no'): kategorični
Age - starost klienta: številski
Gender - spol klienta ('F'-ženski, 'M'-moški): kategorični
Family_Members - število vzdrževanih družinskih članov klienta: številski
Education - dosežena stopnja izobrazbe klienta ('Graduate','High School','Uneducated','College','Unknown'): kategorični
Status - kakšen je status razmerja klienta ('Married', 'Single', 'Divorced', 'Unknown'): kategorični
Income_Category - v katero kategorijo sodi klient glede na prihodek ('average', 'very low', 'high', 'low', 'very high', 'Unknown'): kategorični
Card_Category - kateri tip kartice klient uporablja ('Blue', 'Silver', 'Gold', 'Platinum'): kategorični
Member - koliko mesecev je že klient pri izbrani banki: številski
NumServices - koliko različnih storitev banke klient uporablja: številski
Months_Inactive - koliko mesecev v zadnjem letu je bil klient neaktiven: številski
Contacts_Count - kolikokrat je klient v zadnjem letu kontaktiral banko: številski
Credit_Limit - kakšen limit ima klient na kreditni kartici: številski
MinMonthBalance - minimalno mesečno stanje klienta: številski
ChangeInTransaction - sprememba v velikosti transakcij med začetkom in koncem leta: številski
TransAmount - skupna vsota transakcij v zadnjem letu: številski
TransCount - število transakcij opravljenih v zanjem letu: številski
 Izpit iz vaj (11. 12. 2023)"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

"""
V dataframe preberite vse tri datoteke s podatki: clientInfo.txt , clientActivity.xlsx in clientTransactions.csv . Vse prebrane podatke iz
datotek nato združite v en dataframe, glede na ID klienta. Indeks stolpec naj bo poimenovan ID.
Izpišite zadnjih 5 vrstic tega združenega datafram-a.
Izpišite koliko stolpcev in vrstic je v združenem datafram-u.
Izpišite vse podatke klienta, ki ima ID 714171108.
Izpišite opisno statistiko za vse številske stolpce.
Izpišite prve štiri vrednosti stolpcev za tretjo vrstico v datafram-u.
 Naloga 1 (10 T) 4 (10)
"""


df_excell=pd.read_excel("clientActivity.xlsx")
df_csv=pd.read_csv("clientTransactions.csv", sep=";")
df_txt=pd.read_csv("clientInfo.txt", sep=" ")

df=pd.merge(df_txt,df_csv,on="ID")
df=pd.merge(df,df_excell,on="ID")
print(df.tail(5))
print(df.shape)
#print(np.where(df.ID==714171108))
df.set_index('ID', inplace=True)
specific_id = 714171108
print("\nData for client with ID {}:".format(specific_id))
print(df.loc[specific_id])
"""

print(df.loc[772366833])
print(df.loc[710638233])
print(df.loc[716506083])
print(df.loc[717406983])
print(df.loc[714337233])
"""
print("--------------------------------------------------")
print(df.describe())


print("--------------------------------------------------")
print("\nFirst four values of columns for the third row:")
print(df.iloc[2].head(4))
print("--------------------------------------------------")
"""
Kakšen je povprečen limit klientov na kreditni kartici glede na število družinskih članov? 
Povprečen limit zaokrožite na eno decimalko in ga
sortirajte od največjega do najmanjšega.

V obliki razpredelnice izpišite spol, starost in stopnjo izobrazbe za vse kliente, 
ki imajo odprti račun ('Accaunt_Open') in so v zanjem letu
opravili več kot 12 transakcij

Izrišite graf raztrosa (brez črte), ki bo prikazoval starost klienta glede 
na poljubno vrednost (izmed stolpcev sami izberite vrednost, ki bo
dala smiseln rezultat),

 ločeno po statusu razmerja klienta. Za vsak status 
naj bo prikazan ločen podgraf.

Izrišite graf korelacij med vsemi številskimi stolpci.

Izrišite graf, ki bo prikazoval koliko klientov sodi v katero kategorijo kartice (Card_Category).
 Naloga 2 (20 T) 5+0+3+2+2->12
-1 pr zadni ker s postoti
"""
avg_limit = pd.DataFrame()
avg_limit["limit"] = df["Credit_Limit"]
avg_limit["dr_clani"] = df["Family_Members"]

print(avg_limit.groupby("dr_clani")["limit"].mean().round(1).sort_values(ascending=False))

print("-------------------------------------------------")
filtered_clients = df[(df['Accaunt_Open'] == 'yes') & (df['TransCount'] > 12)]
selected_columns = filtered_clients[['Gender', 'Age', 'Education']]
print(selected_columns)
print("-------------------------------------------------")
g = sns.FacetGrid(df, col="Status")
g.map(plt.scatter, "Age", "Member")
plt.show()
print("-------------------------------------------------")

stevilski = df.select_dtypes(include=["int", "float"]).columns
df_stevilski = pd.DataFrame()
df_stevilski[stevilski] = df[stevilski]
print(df_stevilski)

correlation = df_stevilski.corr()
sns.heatmap(correlation)
plt.show()
print("-------------------------------------------------")
card_category_counts = df['Card_Category'].value_counts()
sns.barplot(x=card_category_counts.index, y=card_category_counts.values)
plt.show()




"""
Izpišite koliko je manjkajočih podatkov v posameznih stolpcih.

Nato manjkajoče podatke iz stolpcev zapolnite s sledečo strategijo:
Contacts_Count zapolnite s povprečno vrednostjo stolpca
Card_Category zapolnite z najpogosteje pojavljeno vrednostjo stolpca
Credit_Limit zapolnite z vrednostjo 0

Ponovno izpišite koliko je manjkajočih vrednosti samo za te tri stolpce.
 Naloga 3 (5 T) 5
"""
nan_df=df.isna()
print(nan_df)
nan=df.isnull().values.sum()
print(nan)

df["Contacts_Count"]=df["Contacts_Count"].fillna(df["Contacts_Count"].mean())
nan=df.isnull().values.sum()
print(nan)
print("------------------------------------",df["Card_Category"].value_counts().idxmax())
df["Card_Category"]=df["Card_Category"].fillna(df["Card_Category"].value_counts().idxmax())

df["Credit_Limit"]=df["Credit_Limit"].fillna(0)

print(df["Contacts_Count"].isnull().values.sum())
print(df["Card_Category"].isnull().values.sum())
print(df["Credit_Limit"].isnull().values.sum())


"""
Ustvarite dve kopiji datafram-a dfRegresija in dfKlasifikacija :
dfKlasifikacija je dataframe, ki ga boste uporabili za klasifikacijo, 
in sicer boste napovedovali ali ima klient pri banki še odprt
račun (Accaunt_Open).

dfRegresija je dataframe, ki ga boste uporabili za regresijo, 
in sicer boste napovedovali število transakcij (TransCount).
Podatke v obeh dataframih ustrezno predprocesirajte(!) -
 kategorične vrednosti pretvorite v dummy vrednosti, številske pa
standardizirajte.

Izpišite prvih 5 vrstic iz vsakega dataframa.
 Naloga 4 (10 T) 10


"""
print("------------------------------------")
stevilski = df.select_dtypes(include=["int", "float"]).columns
scaler=StandardScaler()

dfRegresija=df
dfRegresija=pd.get_dummies(dfRegresija)
dfRegresija[stevilski]=scaler.fit_transform(dfRegresija[stevilski])
print(dfRegresija.head(5))

dfKlasifikacija=df
#print(dfKlasifikacija.columns)
dfKlasifikacija=dfKlasifikacija.drop(columns="Accaunt_Open")
dfKlasifikacija=pd.get_dummies(dfRegresija)
dfKlasifikacija[stevilski]=scaler.fit_transform(dfKlasifikacija[stevilski])
dfKlasifikacija["Accaunt_Open"]=df["Accaunt_Open"]
print(dfKlasifikacija.head(5))


"""
Kot prvo boste s pomočjo klasifikacije poskušali napovedati ali bo nekdo pustil odprt račun 
ali ga bo zaprl (uporabite podatke izn dfKlasifikacija ). 
Izhodni podatek je tako znan, iz vhodnih podatkov pa odstranite še 'MinMonthBalance'
 in 'Credit_Limit'. 
 
 Ker ne vemo kateri
klasifikator bi bil tu najboljši preizkusite tri: naključni gozd,
 k-najbližjih sosedov in odločitveno drevo.
 
Podatke razdelite na učno in testno množico s pomočjo 
stratificirane navzkrižne validacije s sedmimi rezi. Za vsak algoritem merite točnost
njegove napovedi.

Izrišite graf, ki bo pregledno prikazoval povprečno točnost posameznega klasifikatorja.
 Naloga 5 (12 T) 10
"""
#print(dfKlasifikacija.columns)
train=dfKlasifikacija.drop(columns=["Accaunt_Open","MinMonthBalance","Credit_Limit"])
test=dfKlasifikacija["Accaunt_Open"]
klasifikatorji={
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}
skf = StratifiedKFold(n_splits=7)
accuracies = {}

iskanje=GridSearchCV(RandomForestClassifier(),param_grid=klasifikatorji,cv=skf,scoring="accuracy")



for name, clf in klasifikatorji.items():
    scores = cross_val_score(clf, train, test, cv=skf, scoring='accuracy')
    accuracies[name] = scores.mean()
    print(f"{name}: {accuracies[name]}")


plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Classifier')
plt.ylabel('Average Accuracy')
plt.title('Comparison of Classifier Accuracies')
plt.show()

"""
S pomočjo regresija poskusite napovedati število transakcij, ki jih bo klient opravil v enem letu (TransCount), za podatke pa uporabite dataframe
dfRegresija . Iz vhodnih podatkov izpustite tudi podatek NumServices. Podatke delite na učno in testno množico v razmerju 70:30. Naključno
stanje naj bo 123. Za regresor uporabite odločitveno regresijsko drevo.
Kako dobro se je naučil model ocenite s povprečno absolutno napako.
 Naloga 6 (10 T) 10

"""
train=dfRegresija.drop(columns=["TransCount","NumServices"])
test=dfRegresija["TransCount"]

x_train, x_test,y_train,y_test=train_test_split(train,test,test_size=0.3,random_state=123)
DTR=DecisionTreeRegressor()
DTR.fit(x_train,y_train)
napoved=DTR.predict(x_test)

print(mean_absolute_error(napoved,y_test))






"""
Za konec naredite še gručenje nad enakim datasetom, kot ste ga uporabili za regresijo. Podatke transformirajte s pomočjo PCA dekompozicije.
Kot algoritem gručenja uporabite Birch. Podatke delite na 3 gruče.
Izrišite graf, v katerem prikažete transformirane podatke, ki so obarvani glede na gručo, v katero so razvrščeni.
 Naloga 7 (8 T) 0

"""




features = dfRegresija.drop(columns=[ "TransCount", "NumServices"])

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Apply Birch clustering
birch = Birch(n_clusters=3)
clusters = birch.fit_predict(features_pca)

# Plotting
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')

plt.show()




























