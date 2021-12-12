from linearReg import linearReg

import wbdata

indicator_x = {"SL.TLF.TOTL.FE.ZS": "female_labor_force"}
indicators_x = {"SL.TLF.TOTL.FE.ZS": "female_labor_force",'SE.ENR.PRIM.FM.ZS': "School_enrollment_GPI"}
indicator_y = {"SP.ADO.TFRT": "adolescent_fertility_rate"}

X1 = wbdata.get_dataframe(indicator_x, country="TUR", convert_date=True)
X2 = wbdata.get_dataframe(indicators_x, country="TUR", convert_date=True)
y = wbdata.get_dataframe(indicator_y, country="TUR", convert_date=True)

with open('hw2.csv', 'w') as f:
    linearReg(X1,y).to_csv(f)
with open('hw2.csv', 'a') as f:
    linearReg(X2,y).to_csv(f)


