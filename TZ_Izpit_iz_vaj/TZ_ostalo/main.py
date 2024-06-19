import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, \
    StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor





"""

Tematika podatkov: Zadovoljstvo potnikov z letom
Med potniki, ki potujejo z letalom, je bila izvedena raziskava z namenom ugotoviti, kateri dejavniki močno vplivajo na njihovo
zadovoljstvo oziroma nezadovoljstvo z letom. Pridobljeni podatki so razdeljeni v tri datoteke:
datoteko s podatki o potniku in letu ( passengers.csv ),
datoteko s podatki o zadovoljstvu glede na različne kriterije ( passengers_service_rating.txt ),
datoteko s podatki o zamudah in končni oceni zadovoljstva ( passengers_satisfaction.xlsx ).
Opis spremenljivk
Gender - spol potnika (kategorični: 'Female', 'Male')
Customer Type - tip stranke - ali gre za redno stranko ali ne (kategorični: 'Loyal customer', 'disloyal customer')
Age - točna starost potnika (številski)
Type of Travel - namen potnikovega leta (kategorični: 'Personal Travel', 'Business Travel')
Class - v katerem razredu je letel potnik (kategorični: 'Business', 'Eco', 'Eco Plus')
Flight Distance - dolžina leta (številski)
Inflight wifi service - stopnja zadovoljstva za wifi storitev na letalu (številski: 0:'Not Applicable'; 1-5)
Departure/Arrival time convenient - stopnja zadovoljstva za organiziran čas odhoda/prihoda (številski)
Ease of Online booking - stopnja zadovoljstva za spletno rezervacijo (številski)
Gate location - stopnja zadovoljstva za lokacijo vrat, za vkrcanje na letalo (številski)
Food and drink - stopnja zadovoljstva za hrano in pijačo na letalu (številski)
Online boarding - stopnja zadovoljstva za vkrcavanje na letalo (številski)
Seat comfort - stopnja zadovoljstva za udobje sedežev (številski)
Inflight entertainment - stopnja zadovoljstva za ponujeno zabavo med letom (številski)
On-board service - stopnja zadovoljstva za ponujene storitve na letalu (številski)
Leg room service - stopnja zadovoljstva za storitve za noge (številski)
Baggage handling - stopnja zadovoljstva za ravnanje s prtljago (številski)
Checkin service - stopnja zadovoljstva za storitev prijave (številski)
Inflight service - stopnja zadovoljstva za izvedene storitve na letalu (številski)
Cleanliness - stopnja zadovoljstva s čistočo (številski)
Departure Delay in Minutes - čas zamude od odhodu v minutah (številski)
Arrival Delay in Minutes - čas zamude ob prihodu v minutah (številski)
Satisfaction - zadovoljstvo z letom (kategorični: 'Satisfaction', 'neutral or dissatisfaction')
Final_rating - končna ocena zadovoljstva potnika z letom (številski)
 Izpit iz vaj (12. 12. 2023)
"""

"""
Najprej preberite vse tri datoteke s podatki: passengers.csv , passengers_service_rating.txt in passengers_satisfaction.xlsx .
Vse prebrane podatke združite v eno datoteko. Pri združevanju pazite, da potnike združite po njihovi ID številki. Indeks stolpec naj
bo poimenovan Passenger_id .
Izpišite prve štiri vrstice združenih podatkov.
Izpišite število stolpcev ter število vrstic združenih podatkov.
Izpišite podatkovne tipe za vse stolpce.
 Naloga 1 (5T)"""
pasenger_csv=pd.read_csv("passengers.csv", sep=";")
pasenger_txt=pd.read_csv("passengers_service_rating.txt", sep="	")
pasenger_excel=pd.read_excel("passengers_satisfaction.xlsx")

print(pasenger_txt)
print(pasenger_csv)
print(pasenger_excel)

df=pd.merge(pasenger_excel, pasenger_csv,on="passenger_id")
df=pd.merge(df,pasenger_txt,on="passenger_id")
print(df)

print(df.head(4))#Izpišite prve štiri vrstice združenih podatkov
print(df.shape)#Izpišite število stolpcev ter število vrstic združenih podatkov.
#print(df.describe())
print(df.dtypes)#Izpišite podatkovne tipe za vse stolpce.

df.set_index("passenger_id",inplace=True)




"""
Prikažite povprečno dolžino leta ( Flight Distance ), glede na potnikov spol, zaokroženo na eno decimalko.
V obliko razpredelnice shranite starost, potovalni razred in končno oceno leta za vse potnike, ki z letom niso bili zadovoljni in
je bila njihova končna ocena med 11 in 14. Zbrane potnike sortirajte po starosti, od najmlajšega do najstarejšega. Iz te
razpredelnice izpišite podatke za 5 najstarejših potnikov.
Izrišite graf, ki bo prikazoval maksimalni čas zamude ob prihodu ( Arrival Delay in Minutes ) glede na tip zadovoljstva z
letom ( Satisfaction ). Graf naj bo pobarvan v odtenke oranžne.
Izrišite graf raztrosa, tako da bo prikazoval stopnjo zadovoljstva s hrano in pijačo glede na dolžino leta, ločeno glede na
razred ( Class ), v katerem je potnik letel (vsak razred mora biti prikazan v obliki ločenega podgrafa brez črte).
Naloga 2 (15T) 
print(Flight_dist["Flight Distance"]("spol").groupby("spol").mean())


"""
Flight_dist=pd.DataFrame()
Flight_dist["Flight Distance"]=df["Flight Distance"]
Flight_dist["Gender"]=df["Gender"]
#print(Flight_dist["Gender"]("Flight Distance").groupby("Gender").mean())

razpredelnica=df[(df["Satisfaction"]=="dissatisfaction"& 11< df["Final_rating"]<14)]
razp=pd.DataFrame()
razp["Age"]=razpredelnica["Age"]
razp["Class"]=razpredelnica["Class"]
razp["Final_rating"]=razpredelnica["Final_rating"]
print(razp.groupby(razp["Age"]).head(5))

"""
Izpišite koliko je manjkajočih podatkov v posameznih stolpcih.
Nato manjkajoče podatke iz stolpcev zapolnite s sledečo strategijo:
Flight Distance in Arrival Delay in Minutes zapolnite s povprečno vrednostjo stolpca.
Type of Travel in Class zapolnite z najpogosteje pojavljeno vrednostjo stolpca
Za ostale manjkajoče vrednosti poskrbite tako, vrstice z manjkajočimi podatki odstranite.
Ponovno izpišite število manjkajočih podatkov, vendar samo za stoplce, ki smo jih dopolnjevali.
 Naloga 3 (5 T)
"""
mankajoce=df.isnull().values.sum()
print(mankajoce)#Izpišite koliko je manjkajočih podatkov v posameznih stolpcih.

df["Flight Distance"]=df["Flight Distance"].fillna(df["Flight Distance"].mean())
mankajoce=df.isnull().values.sum()
print(mankajoce)
df["Arrival Delay in Minutes"]=df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].mean())
mankajoce=df.isnull().values.sum()
print(mankajoce)#Flight Distance in Arrival Delay in Minutes zapolnite s povprečno vrednostjo stolpca.

#df["Type of Travel"]=df["Type of Travel"].fillna(df["Type of Travel"].values.argmax())

#df=df.drop(df.isna)


df=df.dropna()
print(df["Flight Distance"].isnull().values.sum())
print(df["Arrival Delay in Minutes"].isnull().values.sum())
print(df["Type of Travel"].isnull().values.sum())
print(df["Class"].isnull().values.sum())








"""
Ustvarite dve kopiji datafram-a dfRegresija in dfKlasifikacija :
dfKlasifikacija je dataframe, ki ga boste uporabili za klasifikacijo,
 in sicer boste napovedovali ali je bil potnik
zadovoljen z letom ( Satisfaction ).

dfRegresija je dataframe, ki ga boste uporabili za regresijo,
 in sicer boste napovedovali končna ocena zadovoljstva
potnika z letom ( Final_rating ).

Podatke v obeh dataframih ustrezno predprocesirajte(!) 
- kategorične vrednosti pretvorite v številske (lahko uporabite
LabelEncoder), številske pa standardizirajte.
Izpišite prve 3 vrstice iz vsakega dataframa.
 Naloga 4 (10 T)
"""
le=LabelEncoder()
scale=StandardScaler()
stev=df.select_dtypes(include=["int64","float64"]).columns
kategoricni=df.select_dtypes(include=["object"]).columns

dfRegresija=df
dfRegresija=dfRegresija.drop(columns="Final_rating")
print(dfRegresija.shape)
stev=stev.drop("Final_rating")
dfRegresija[kategoricni]=le.fit_transform(kategoricni)
print(dfRegresija)
dfRegresija[stev]=scale.fit_transform(dfRegresija[stev])
dfRegresija["Final_rating"]=df["Final_rating"]


print(dfRegresija.shape)





dfKlasifikacija=df
dfKlasifikacija[kategoricni]=le.fit_transform(kategoricni)
dfKlasifikacija[stev]=scale.fit_transform(dfKlasifikacija[stev])

print("Klasifikacija: ", dfKlasifikacija.head(3))
print("Regresija: ", dfRegresija.head(3))
"""
S pomočjo regresija poskusite napovedati končno oceno zadovoljstva potnika z letom ( Final_rating ). Za podatke uporabite
predprocesiran dataframe dfRegresija . Iz vhodnih podatkov izpustite tudi podatek Satisfaction . Podatke delite na učno in
testno množico v razmerju 75:25. Naključno stanje naj bo 321. Za regresor uporabite regresijsko drevo.
Kako dobro se je naučil model ocenite s povprečno kvadratno napako, zaokroženo na eno decimalko.
Naloga 5 (10 T) 
"""
dtr=DecisionTreeRegressor()
test=dfRegresija["Final_rating"]
train=dfRegresija.drop(columns=["Satisfaction","Final_rating"])

x_train, x_test, y_train, y_test=train_test_split(train,test,test_size=0.25,random_state=321)

dtr.fit(x_train,y_train)
x_test=dtr.predict(x_test)

#print("tocnost: ",dtr.score(x_test,y_test).round(1)
"""
ValueError: Expected 2D array, got 1D array instead:
array=[100.  59.  94. ...  58.  85.  24.].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
"""




"""
S pomočjo klasifikacije bomo napovedovali zadovoljstvo potnikov z letom ( Satisfaction ). Iz vhodnih podatkov odstranite še
stolpec Final_rating . Podatke iz predprocesiranega dfKlasifikacija delite na učne in testne in sicer s pomočjo stratificirane
delitve na 3 folde.
Nad podatkih preizkusite dva klasifikatorja - naključni gozd in K najbližjih sosedov. Ker želimo doseči najvišjo možno točnost
klasifikacije to izvedite s pomočjo iskanja najboljših nastavitev parametrov po principu mreže (GridSearchCV).
Za nakjučni gozd preizkusite:
število dreves 50 in 100,
kriterij "gini" in "entropy".
Za K najbližjih sosedov pa:
3, 5 in 10 sosedov.
Najboljše izračunane vrednosti točnosti za oba klasifkatorja prikažite v stolpičnem grafu.
 Naloga 6 (20 T)
"""
test=dfKlasifikacija["Satisfaction"]
train=dfKlasifikacija.drop(columns=["Satisfaction","Final_rating"])
spl=StratifiedShuffleSplit(n_splits=3)

RFC=RandomForestClassifier()
KNN=KNeighborsClassifier()
drevesa={

"criterion":["gini","entropy"],


}
knn={"n_estimatorsint":[50,100],
    "n_neighbors":[3,5,10]}

iskanje=GridSearchCV(KNeighborsClassifier,param_grid=knn,scoring="accuracy",cv=spl)
iskanje.fit(train,test)
drevo_best=iskanje.get_params
iskanje=GridSearchCV(RandomForestClassifier,param_grid=drevesa,scoring="accuracy",cv=spl)
knn_best=iskanje.get_params
iskanje.fit(train,test)
print(drevo_best)
print(knn_best)


#TypeError: Cannot clone object. You should provide an instance of scikit-learn estimator instead of a class


"""
Za konec naredite še gručenje nad enakim datasetom, kot ste ga uporabili za regresijo. Podatke transformirajte s pomočjo PCA
dekompozicije. Kot algoritem gručenja uporabite KMeans.
Da boste vedeli koliko je najbolj optimalno število gruč na katere je smiselno deliti podatke pred gručenjem izrišite graf z
izračunanimi ineciami za od 1 do 7 gruč, nad transformiranimi poatki. Po pravilu komolca iz grafa preberite najbolj optimalno
število gruč in ga uporabite v algoritmu.
Izrišite graf, v katerem prikažete transformirane podatke, ki so obarvani glede na gručo, v katero so razvrščeni.
 Naloga 7 (10 T)
"""































































