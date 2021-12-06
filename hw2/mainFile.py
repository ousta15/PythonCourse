from linearReg import linearReg
import numpy as np
import wbdata

indicators_x = {"SL.TLF.TOTL.FE.ZS": "female_labor_force", "SP.ADO.TFRT": "adolescent_fertility_rate"}
indicator_y = {'SE.ENR.PRIM.FM.ZS': "School_enrollment_GPI"}
X = wbdata.get_dataframe(indicators_x, country="TUR", convert_date=True)
y = wbdata.get_dataframe(indicators_x, country="TUR", convert_date=True)



b = linearReg(X,y)

print(b)
