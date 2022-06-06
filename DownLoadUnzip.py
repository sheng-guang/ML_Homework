import urllib.request

doq = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
pat_sav = "./AirQualityUCI.zip"
urllib.request.urlretrieve(doq, pat_sav)
# urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip")


import zipfile

zip = zipfile.ZipFile('./AirQualityUCI.zip', 'r')
for name in zip.namelist():
    zip.extract(name, '.')
# The .csv version of the file has some typing issues, so we use the excel version
