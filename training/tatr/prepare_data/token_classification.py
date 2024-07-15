import spacy
 
nlp = spacy.load("en_core_web_sm")

text = "Contract 50254662-1 Product Description / Product Code SPS - EE - RNW - AG - 100-250 Protection Suite Enterprise Edition , Renewal Software Maintenance , ACD - GOV 100-249 Devices 1 YR Covered Product : SPS - EE - ADD - 1-25 Protection Suite Enterprise Edition , Additional Quantity License . 1-24 Devices Quantity Shipped 1 Unit Price 1,069.45 Extended Price 1,069.45 Tax Description US - NO TAX STATES Total Item Tax 0.00"
doc = nlp(text)
 
print(nlp.pipe_names)