from linearReg import linearReg

import wbdata

indicator_x = {"SL.TLF.TOTL.FE.ZS": "female_labor_force"}
indicators_x = {"SL.TLF.TOTL.FE.ZS": "female_labor_force",'SE.ENR.PRIM.FM.ZS': "School_enrollment_GPI"}
indicator_y = {"SP.ADO.TFRT": "adolescent_fertility_rate"}
X1 = wbdata.get_dataframe(indicator_x, country="TUR", convert_date=True)
X2 = wbdata.get_dataframe(indicators_x, country="TUR", convert_date=True)
y = wbdata.get_dataframe(indicator_y, country="TUR", convert_date=True)

linearReg(X1,y).to_csv("hw2.csv")
linearReg(X2,y).to_csv("hw2.csv")


