# Downloading Dataset from Google Drive

str1="1G1Kd3jd3dobgN_N0tu3tXKrMEescmmIw"
str2="1Oat2lAZUF-uQY1iZ60p8roGVKGSdgCwN"
str3="1rtUdnYqEQR_Usw05zfOkOIDjXLbe2WAt"
str4="1G-swRizJR6Xya8kqahKNEh8HMdG7PZV-"
str5="16BjZM192Go2pkufcxr51lxBHdJfKYaMt"
str6="1Nl8bt9D4ScSQI9yKP6ohBioAmR7e2JLe"
str7="1zjjjlSpmW6tZzg0l3Yiq2-GYWce37xeG"
str8="1OVzD6_XAifMPt1tMiU5mgkkRrTl2Lln0"

data_link=( ${str1} ${str2} ${str3} ${str4} ${str5} ${str6} ${str7} ${str8})

for link in ${data_link[@]} 
    do
        echo "Downloading ${link}"
        gdown --id ${link}
    done